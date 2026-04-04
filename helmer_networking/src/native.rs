#![cfg(not(target_arch = "wasm32"))]

use std::{
    collections::{HashMap, VecDeque},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use futures_util::{SinkExt, StreamExt};
use tokio::{
    net::{TcpListener, UdpSocket},
    runtime::Builder,
    sync::mpsc,
    time::sleep,
};
use tokio_tungstenite::{accept_async, connect_async, tungstenite::Message};

use crate::{
    NetworkClientConfig, NetworkConnectionId, NetworkEvent, NetworkLane, NetworkServerConfig,
    NetworkTransportStats,
    hub::push_event,
    wire::{
        DatagramWire, PacketWire, ReliableWireFrame, TransportHello, decode_datagram,
        decode_reliable_frame, encode_datagram, encode_reliable_frame,
    },
};

#[derive(Debug)]
pub(crate) enum BackendCommand {
    Listen(NetworkServerConfig),
    Connect(NetworkClientConfig),
    Send {
        connection_id: NetworkConnectionId,
        lane: NetworkLane,
        payload: Vec<u8>,
    },
    Disconnect {
        connection_id: NetworkConnectionId,
    },
    Shutdown,
}

#[derive(Clone, Default)]
pub(crate) struct NativeTelemetryMap {
    inner: Arc<Mutex<HashMap<NetworkConnectionId, Arc<ConnectionTelemetry>>>>,
}

impl NativeTelemetryMap {
    fn insert(&self, connection_id: NetworkConnectionId, telemetry: Arc<ConnectionTelemetry>) {
        self.inner
            .lock()
            .expect("network telemetry poisoned")
            .insert(connection_id, telemetry);
    }

    fn remove(&self, connection_id: NetworkConnectionId) {
        self.inner
            .lock()
            .expect("network telemetry poisoned")
            .remove(&connection_id);
    }

    pub(crate) fn snapshot(
        &self,
        connection_id: NetworkConnectionId,
    ) -> Option<NetworkTransportStats> {
        self.inner
            .lock()
            .expect("network telemetry poisoned")
            .get(&connection_id)
            .map(|telemetry| telemetry.snapshot())
    }
}

#[derive(Default)]
struct ConnectionTelemetry {
    queued_reliable_frames: AtomicU64,
    queued_reliable_bytes: AtomicU64,
    smoothed_rtt_millis: AtomicU64,
}

impl ConnectionTelemetry {
    fn snapshot(&self) -> NetworkTransportStats {
        let rtt = self.smoothed_rtt_millis.load(Ordering::Relaxed);
        NetworkTransportStats {
            queued_reliable_frames: self.queued_reliable_frames.load(Ordering::Relaxed) as usize,
            queued_reliable_bytes: self.queued_reliable_bytes.load(Ordering::Relaxed),
            smoothed_rtt_millis: (rtt != 0).then_some(rtt as u32),
        }
    }

    fn note_rtt_sample(&self, sample_millis: u32) {
        let sample_millis = sample_millis.max(1);
        let mut current = self.smoothed_rtt_millis.load(Ordering::Relaxed);
        loop {
            let next = if current == 0 {
                u64::from(sample_millis)
            } else {
                (current.saturating_mul(7) + u64::from(sample_millis)) / 8
            };
            match self.smoothed_rtt_millis.compare_exchange(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(updated) => current = updated,
            }
        }
    }
}

struct QueuedReliableFrame {
    bytes: Vec<u8>,
    byte_len: u64,
}

pub(crate) fn spawn_native_backend(
    events: Arc<Mutex<VecDeque<NetworkEvent>>>,
    telemetry: NativeTelemetryMap,
) -> mpsc::UnboundedSender<BackendCommand> {
    let (command_tx, command_rx) = mpsc::unbounded_channel();
    std::thread::Builder::new()
        .name("helmer-networking".to_owned())
        .spawn(move || {
            let runtime = Builder::new_multi_thread()
                .enable_all()
                .thread_name("helmer-networking-io")
                .build()
                .expect("failed to build helmer_networking runtime");
            runtime.block_on(async move {
                NativeDriver::new(events, telemetry).run(command_rx).await;
            });
        })
        .expect("failed to spawn helmer_networking backend");
    command_tx
}

fn current_time_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn note_reliable_enqueue(telemetry: &ConnectionTelemetry, byte_len: u64) {
    telemetry
        .queued_reliable_frames
        .fetch_add(1, Ordering::Relaxed);
    telemetry
        .queued_reliable_bytes
        .fetch_add(byte_len, Ordering::Relaxed);
}

fn note_reliable_dequeue(telemetry: &ConnectionTelemetry, byte_len: u64) {
    telemetry
        .queued_reliable_frames
        .fetch_sub(1, Ordering::Relaxed);
    telemetry
        .queued_reliable_bytes
        .fetch_sub(byte_len, Ordering::Relaxed);
}

fn enqueue_reliable_frame(
    tx: &mpsc::UnboundedSender<QueuedReliableFrame>,
    telemetry: &Arc<ConnectionTelemetry>,
    frame: ReliableWireFrame,
) -> bool {
    let bytes = match encode_reliable_frame(&frame) {
        Ok(bytes) => bytes,
        Err(_) => return false,
    };
    let queued = QueuedReliableFrame {
        byte_len: bytes.len() as u64,
        bytes,
    };
    note_reliable_enqueue(telemetry, queued.byte_len);
    let byte_len = queued.byte_len;
    if tx.send(queued).is_err() {
        note_reliable_dequeue(telemetry, byte_len);
        return false;
    }
    true
}

fn spawn_reliable_ping_task(
    tx: mpsc::UnboundedSender<QueuedReliableFrame>,
    telemetry: Arc<ConnectionTelemetry>,
) {
    tokio::spawn(async move {
        loop {
            sleep(Duration::from_millis(500)).await;
            if !enqueue_reliable_frame(
                &tx,
                &telemetry,
                ReliableWireFrame::Ping {
                    sent_at_millis: current_time_millis(),
                },
            ) {
                break;
            }
        }
    });
}

struct NativeDriver {
    events: Arc<Mutex<VecDeque<NetworkEvent>>>,
    telemetry: NativeTelemetryMap,
    mode: DriverMode,
}

enum DriverMode {
    Idle,
    Server(ServerRuntime),
    Client(ClientRuntime),
}

impl NativeDriver {
    fn new(events: Arc<Mutex<VecDeque<NetworkEvent>>>, telemetry: NativeTelemetryMap) -> Self {
        Self {
            events,
            telemetry,
            mode: DriverMode::Idle,
        }
    }

    async fn refresh_mode(&mut self) {
        let should_reset = match &self.mode {
            DriverMode::Client(client) => !client.is_active().await,
            DriverMode::Idle | DriverMode::Server(_) => false,
        };
        if should_reset {
            self.mode = DriverMode::Idle;
        }
    }

    async fn run(&mut self, mut commands: mpsc::UnboundedReceiver<BackendCommand>) {
        while let Some(command) = commands.recv().await {
            self.refresh_mode().await;
            match command {
                BackendCommand::Listen(config) => {
                    if !matches!(self.mode, DriverMode::Idle) {
                        push_event(
                            &self.events,
                            NetworkEvent::Error {
                                context: "listen".to_owned(),
                                message: "backend already initialized".to_owned(),
                            },
                        );
                        continue;
                    }
                    match ServerRuntime::start(
                        config,
                        Arc::clone(&self.events),
                        self.telemetry.clone(),
                    )
                    .await
                    {
                        Ok(server) => self.mode = DriverMode::Server(server),
                        Err(err) => push_event(
                            &self.events,
                            NetworkEvent::Error {
                                context: "listen".to_owned(),
                                message: err,
                            },
                        ),
                    }
                }
                BackendCommand::Connect(config) => {
                    if !matches!(self.mode, DriverMode::Idle) {
                        push_event(
                            &self.events,
                            NetworkEvent::Error {
                                context: "connect".to_owned(),
                                message: "backend already initialized".to_owned(),
                            },
                        );
                        continue;
                    }
                    let client =
                        ClientRuntime::new(Arc::clone(&self.events), self.telemetry.clone());
                    if client.connect(config).await {
                        self.mode = DriverMode::Client(client);
                    }
                }
                BackendCommand::Send {
                    connection_id,
                    lane,
                    payload,
                } => match &self.mode {
                    DriverMode::Server(server) => {
                        server.send(connection_id, lane, payload).await;
                    }
                    DriverMode::Client(client) => {
                        client.send(connection_id, lane, payload).await;
                    }
                    DriverMode::Idle => push_event(
                        &self.events,
                        NetworkEvent::Error {
                            context: "send".to_owned(),
                            message: "backend not initialized".to_owned(),
                        },
                    ),
                },
                BackendCommand::Disconnect { connection_id } => match &self.mode {
                    DriverMode::Server(server) => server.disconnect(connection_id).await,
                    DriverMode::Client(client) => client.disconnect(connection_id).await,
                    DriverMode::Idle => {}
                },
                BackendCommand::Shutdown => break,
            }
        }
    }
}

struct ServerRuntime {
    peers: Arc<tokio::sync::Mutex<HashMap<NetworkConnectionId, ServerPeerState>>>,
    udp_socket: Option<Arc<UdpSocket>>,
}

#[derive(Clone)]
struct ServerPeerState {
    reliable_tx: mpsc::UnboundedSender<QueuedReliableFrame>,
    telemetry: Arc<ConnectionTelemetry>,
    udp_peer: Option<std::net::SocketAddr>,
    udp_token: u64,
    next_unreliable_sequence: u64,
    last_unreliable_sequence: u64,
}

impl ServerRuntime {
    async fn start(
        config: NetworkServerConfig,
        events: Arc<Mutex<VecDeque<NetworkEvent>>>,
        telemetry: NativeTelemetryMap,
    ) -> Result<Self, String> {
        let listener = TcpListener::bind(&config.reliable_bind)
            .await
            .map_err(|err| format!("failed to bind reliable listener: {err}"))?;
        let reliable_addr = listener
            .local_addr()
            .map_err(|err| format!("failed to query reliable listener address: {err}"))?;

        let udp_socket = match config.unreliable_bind.as_ref() {
            Some(bind) => {
                Some(Arc::new(UdpSocket::bind(bind).await.map_err(|err| {
                    format!("failed to bind unreliable socket: {err}")
                })?))
            }
            None => None,
        };

        let unreliable_addr = match (config.public_unreliable_addr, udp_socket.as_ref()) {
            (Some(addr), _) => Some(addr),
            (None, Some(socket)) => socket.local_addr().ok().map(|addr| addr.to_string()),
            (None, None) => None,
        };

        push_event(
            &events,
            NetworkEvent::Listening {
                reliable_addr: reliable_addr.to_string(),
                unreliable_addr: unreliable_addr.clone(),
            },
        );

        let peers = Arc::new(tokio::sync::Mutex::new(HashMap::new()));
        let next_connection_id = Arc::new(AtomicU64::new(1));

        tokio::spawn(run_server_listener(
            listener,
            Arc::clone(&peers),
            Arc::clone(&events),
            telemetry,
            unreliable_addr,
            next_connection_id,
        ));

        if let Some(socket) = udp_socket.as_ref() {
            tokio::spawn(run_server_udp(
                Arc::clone(socket),
                Arc::clone(&peers),
                Arc::clone(&events),
            ));
        }

        Ok(Self { peers, udp_socket })
    }

    async fn send(&self, connection_id: NetworkConnectionId, lane: NetworkLane, payload: Vec<u8>) {
        let mut peers = self.peers.lock().await;
        let Some(peer) = peers.get_mut(&connection_id) else {
            return;
        };

        if lane == NetworkLane::UnreliableSequenced
            && let (Some(socket), Some(target_addr)) = (self.udp_socket.as_ref(), peer.udp_peer)
        {
            let frame = DatagramWire {
                connection_id,
                token: peer.udp_token,
                sequence: peer.next_unreliable_sequence,
                payload,
            };
            peer.next_unreliable_sequence = peer.next_unreliable_sequence.saturating_add(1);
            let bytes = match encode_datagram(&frame) {
                Ok(bytes) => bytes,
                Err(_) => return,
            };
            let socket = Arc::clone(socket);
            tokio::spawn(async move {
                let _ = socket.send_to(&bytes, target_addr).await;
            });
            return;
        }

        let _ = enqueue_reliable_frame(
            &peer.reliable_tx,
            &peer.telemetry,
            ReliableWireFrame::Payload(PacketWire { lane, payload }),
        );
    }

    async fn disconnect(&self, connection_id: NetworkConnectionId) {
        let mut peers = self.peers.lock().await;
        if let Some(peer) = peers.remove(&connection_id) {
            let _ = enqueue_reliable_frame(
                &peer.reliable_tx,
                &peer.telemetry,
                ReliableWireFrame::Disconnect {
                    reason: "server requested disconnect".to_owned(),
                },
            );
        }
    }
}

async fn run_server_listener(
    listener: TcpListener,
    peers: Arc<tokio::sync::Mutex<HashMap<NetworkConnectionId, ServerPeerState>>>,
    events: Arc<Mutex<VecDeque<NetworkEvent>>>,
    telemetry: NativeTelemetryMap,
    unreliable_addr: Option<String>,
    next_connection_id: Arc<AtomicU64>,
) {
    loop {
        let (stream, _) = match listener.accept().await {
            Ok(value) => value,
            Err(err) => {
                push_event(
                    &events,
                    NetworkEvent::Error {
                        context: "server_accept".to_owned(),
                        message: err.to_string(),
                    },
                );
                break;
            }
        };

        tokio::spawn(handle_server_connection(
            stream,
            Arc::clone(&peers),
            Arc::clone(&events),
            telemetry.clone(),
            unreliable_addr.clone(),
            Arc::clone(&next_connection_id),
        ));
    }
}

async fn handle_server_connection(
    stream: tokio::net::TcpStream,
    peers: Arc<tokio::sync::Mutex<HashMap<NetworkConnectionId, ServerPeerState>>>,
    events: Arc<Mutex<VecDeque<NetworkEvent>>>,
    telemetry_map: NativeTelemetryMap,
    unreliable_addr: Option<String>,
    next_connection_id: Arc<AtomicU64>,
) {
    let ws = match accept_async(stream).await {
        Ok(ws) => ws,
        Err(err) => {
            push_event(
                &events,
                NetworkEvent::Error {
                    context: "server_handshake".to_owned(),
                    message: err.to_string(),
                },
            );
            return;
        }
    };

    let connection_id = NetworkConnectionId(next_connection_id.fetch_add(1, Ordering::Relaxed));
    let udp_token = fastrand::u64(..);
    let unreliable_supported = unreliable_addr.is_some();
    let telemetry = Arc::new(ConnectionTelemetry::default());

    let (mut sink, mut stream) = ws.split();
    let (reliable_tx, mut reliable_rx) = mpsc::unbounded_channel::<QueuedReliableFrame>();
    telemetry_map.insert(connection_id, Arc::clone(&telemetry));

    {
        let mut peers_guard = peers.lock().await;
        peers_guard.insert(
            connection_id,
            ServerPeerState {
                reliable_tx: reliable_tx.clone(),
                telemetry: Arc::clone(&telemetry),
                udp_peer: None,
                udp_token,
                next_unreliable_sequence: 1,
                last_unreliable_sequence: 0,
            },
        );
    }

    let _ = enqueue_reliable_frame(
        &reliable_tx,
        &telemetry,
        ReliableWireFrame::Hello(TransportHello {
            connection_id,
            unreliable_token: unreliable_supported.then_some(udp_token),
            unreliable_addr,
        }),
    );
    spawn_reliable_ping_task(reliable_tx.clone(), Arc::clone(&telemetry));

    push_event(
        &events,
        NetworkEvent::Connected {
            connection_id,
            unreliable_supported,
        },
    );

    let writer_telemetry = Arc::clone(&telemetry);
    let writer = tokio::spawn(async move {
        while let Some(frame) = reliable_rx.recv().await {
            note_reliable_dequeue(&writer_telemetry, frame.byte_len);
            if sink
                .send(Message::Binary(frame.bytes.into()))
                .await
                .is_err()
            {
                break;
            }
        }
    });

    while let Some(message) = stream.next().await {
        let bytes = match message {
            Ok(Message::Binary(bytes)) => bytes,
            Ok(Message::Close(_)) => break,
            Ok(_) => continue,
            Err(err) => {
                push_event(
                    &events,
                    NetworkEvent::Error {
                        context: "server_read".to_owned(),
                        message: err.to_string(),
                    },
                );
                break;
            }
        };

        let frame: ReliableWireFrame = match decode_reliable_frame(&bytes) {
            Ok(frame) => frame,
            Err(err) => {
                push_event(
                    &events,
                    NetworkEvent::Error {
                        context: "server_decode".to_owned(),
                        message: err.to_string(),
                    },
                );
                continue;
            }
        };

        match frame {
            ReliableWireFrame::Payload(packet) => push_event(
                &events,
                NetworkEvent::Packet {
                    connection_id,
                    lane: packet.lane,
                    payload: packet.payload,
                },
            ),
            ReliableWireFrame::Disconnect { reason } => {
                push_event(
                    &events,
                    NetworkEvent::Disconnected {
                        connection_id,
                        reason,
                    },
                );
                break;
            }
            ReliableWireFrame::Ping { sent_at_millis } => {
                let _ = enqueue_reliable_frame(
                    &reliable_tx,
                    &telemetry,
                    ReliableWireFrame::Pong { sent_at_millis },
                );
            }
            ReliableWireFrame::Pong { sent_at_millis } => {
                let elapsed = current_time_millis().saturating_sub(sent_at_millis);
                telemetry.note_rtt_sample(elapsed.min(u64::from(u32::MAX)) as u32);
            }
            ReliableWireFrame::Hello(_) => {}
        }
    }

    writer.abort();
    peers.lock().await.remove(&connection_id);
    telemetry_map.remove(connection_id);
    push_event(
        &events,
        NetworkEvent::Disconnected {
            connection_id,
            reason: "reliable transport closed".to_owned(),
        },
    );
}

async fn run_server_udp(
    socket: Arc<UdpSocket>,
    peers: Arc<tokio::sync::Mutex<HashMap<NetworkConnectionId, ServerPeerState>>>,
    events: Arc<Mutex<VecDeque<NetworkEvent>>>,
) {
    let mut buffer = vec![0_u8; 65_535];
    loop {
        let (len, addr) = match socket.recv_from(&mut buffer).await {
            Ok(value) => value,
            Err(err) => {
                push_event(
                    &events,
                    NetworkEvent::Error {
                        context: "server_udp_recv".to_owned(),
                        message: err.to_string(),
                    },
                );
                break;
            }
        };

        let frame: DatagramWire = match decode_datagram(&buffer[..len]) {
            Ok(frame) => frame,
            Err(err) => {
                push_event(
                    &events,
                    NetworkEvent::Error {
                        context: "server_udp_decode".to_owned(),
                        message: err.to_string(),
                    },
                );
                continue;
            }
        };

        let accepted = {
            let mut peers_guard = peers.lock().await;
            match peers_guard.get_mut(&frame.connection_id) {
                Some(peer)
                    if peer.udp_token == frame.token
                        && frame.sequence > peer.last_unreliable_sequence =>
                {
                    peer.last_unreliable_sequence = frame.sequence;
                    peer.udp_peer = Some(addr);
                    true
                }
                _ => false,
            }
        };

        if accepted {
            push_event(
                &events,
                NetworkEvent::Packet {
                    connection_id: frame.connection_id,
                    lane: NetworkLane::UnreliableSequenced,
                    payload: frame.payload,
                },
            );
        }
    }
}

struct ClientRuntime {
    events: Arc<Mutex<VecDeque<NetworkEvent>>>,
    state: Arc<tokio::sync::Mutex<ClientState>>,
    telemetry: NativeTelemetryMap,
}

#[derive(Default)]
struct ClientState {
    connection_id: Option<NetworkConnectionId>,
    reliable_tx: Option<mpsc::UnboundedSender<QueuedReliableFrame>>,
    telemetry: Option<Arc<ConnectionTelemetry>>,
    udp_socket: Option<Arc<UdpSocket>>,
    udp_remote: Option<std::net::SocketAddr>,
    udp_token: Option<u64>,
    next_unreliable_sequence: u64,
    last_unreliable_sequence: u64,
}

impl ClientRuntime {
    fn new(events: Arc<Mutex<VecDeque<NetworkEvent>>>, telemetry: NativeTelemetryMap) -> Self {
        Self {
            events,
            state: Arc::new(tokio::sync::Mutex::new(ClientState::default())),
            telemetry,
        }
    }

    async fn is_active(&self) -> bool {
        let state = self.state.lock().await;
        state.reliable_tx.is_some() || state.connection_id.is_some()
    }

    async fn connect(&self, config: NetworkClientConfig) -> bool {
        let ws = match connect_async(config.reliable_url.clone()).await {
            Ok((ws, _)) => ws,
            Err(err) => {
                push_event(
                    &self.events,
                    NetworkEvent::Error {
                        context: "client_connect".to_owned(),
                        message: err.to_string(),
                    },
                );
                return false;
            }
        };

        let (mut sink, mut stream) = ws.split();
        let (reliable_tx, mut reliable_rx) = mpsc::unbounded_channel::<QueuedReliableFrame>();
        let telemetry = Arc::new(ConnectionTelemetry::default());

        {
            let mut state = self.state.lock().await;
            state.reliable_tx = Some(reliable_tx.clone());
            state.telemetry = Some(Arc::clone(&telemetry));
        }

        let writer_telemetry = Arc::clone(&telemetry);
        tokio::spawn(async move {
            while let Some(frame) = reliable_rx.recv().await {
                note_reliable_dequeue(&writer_telemetry, frame.byte_len);
                if sink
                    .send(Message::Binary(frame.bytes.into()))
                    .await
                    .is_err()
                {
                    break;
                }
            }
        });
        spawn_reliable_ping_task(reliable_tx.clone(), Arc::clone(&telemetry));

        let state = Arc::clone(&self.state);
        let events = Arc::clone(&self.events);
        let telemetry_map = self.telemetry.clone();
        let read_telemetry = Arc::clone(&telemetry);
        let read_tx = reliable_tx.clone();
        tokio::spawn(async move {
            while let Some(message) = stream.next().await {
                let bytes = match message {
                    Ok(Message::Binary(bytes)) => bytes,
                    Ok(Message::Close(_)) => {
                        let connection_id = clear_client_state(&state, &telemetry_map).await;
                        if let Some(connection_id) = connection_id {
                            push_event(
                                &events,
                                NetworkEvent::Disconnected {
                                    connection_id,
                                    reason: "connection closed".to_owned(),
                                },
                            );
                        }
                        break;
                    }
                    Ok(_) => continue,
                    Err(err) => {
                        let connection_id = clear_client_state(&state, &telemetry_map).await;
                        if let Some(connection_id) = connection_id {
                            push_event(
                                &events,
                                NetworkEvent::Disconnected {
                                    connection_id,
                                    reason: err.to_string(),
                                },
                            );
                        }
                        push_event(
                            &events,
                            NetworkEvent::Error {
                                context: "client_read".to_owned(),
                                message: err.to_string(),
                            },
                        );
                        break;
                    }
                };

                let frame: ReliableWireFrame = match decode_reliable_frame(&bytes) {
                    Ok(frame) => frame,
                    Err(err) => {
                        push_event(
                            &events,
                            NetworkEvent::Error {
                                context: "client_decode".to_owned(),
                                message: err.to_string(),
                            },
                        );
                        continue;
                    }
                };

                match frame {
                    ReliableWireFrame::Hello(hello) => {
                        let mut state_guard = state.lock().await;
                        state_guard.connection_id = Some(hello.connection_id);
                        state_guard.udp_token = hello.unreliable_token;
                        state_guard.udp_remote = hello
                            .unreliable_addr
                            .as_ref()
                            .and_then(|value| value.parse().ok());

                        let can_bind_udp =
                            state_guard.udp_remote.is_some() && state_guard.udp_token.is_some();
                        if can_bind_udp
                            && state_guard.udp_socket.is_none()
                            && let Some(bind_addr) = config.unreliable_bind.as_ref()
                        {
                            match UdpSocket::bind(bind_addr).await {
                                Ok(socket) => {
                                    let socket = Arc::new(socket);
                                    tokio::spawn(run_client_udp(
                                        Arc::clone(&socket),
                                        Arc::clone(&state),
                                        Arc::clone(&events),
                                    ));
                                    state_guard.udp_socket = Some(socket);
                                }
                                Err(err) => push_event(
                                    &events,
                                    NetworkEvent::Error {
                                        context: "client_udp_bind".to_owned(),
                                        message: err.to_string(),
                                    },
                                ),
                            }
                        }
                        telemetry_map.insert(hello.connection_id, Arc::clone(&read_telemetry));

                        push_event(
                            &events,
                            NetworkEvent::Connected {
                                connection_id: hello.connection_id,
                                unreliable_supported: state_guard.udp_socket.is_some()
                                    && state_guard.udp_remote.is_some()
                                    && state_guard.udp_token.is_some(),
                            },
                        );
                    }
                    ReliableWireFrame::Payload(packet) => {
                        let connection_id = {
                            let state_guard = state.lock().await;
                            state_guard.connection_id
                        };
                        if let Some(connection_id) = connection_id {
                            push_event(
                                &events,
                                NetworkEvent::Packet {
                                    connection_id,
                                    lane: packet.lane,
                                    payload: packet.payload,
                                },
                            );
                        }
                    }
                    ReliableWireFrame::Ping { sent_at_millis } => {
                        let _ = enqueue_reliable_frame(
                            &read_tx,
                            &read_telemetry,
                            ReliableWireFrame::Pong { sent_at_millis },
                        );
                    }
                    ReliableWireFrame::Pong { sent_at_millis } => {
                        let elapsed = current_time_millis().saturating_sub(sent_at_millis);
                        read_telemetry.note_rtt_sample(elapsed.min(u64::from(u32::MAX)) as u32);
                    }
                    ReliableWireFrame::Disconnect { reason } => {
                        let connection_id = clear_client_state(&state, &telemetry_map).await;
                        if let Some(connection_id) = connection_id {
                            push_event(
                                &events,
                                NetworkEvent::Disconnected {
                                    connection_id,
                                    reason,
                                },
                            );
                        }
                        break;
                    }
                }
            }
        });
        true
    }

    async fn send(&self, connection_id: NetworkConnectionId, lane: NetworkLane, payload: Vec<u8>) {
        let mut state = self.state.lock().await;
        if state.connection_id != Some(connection_id) {
            return;
        }

        if lane == NetworkLane::UnreliableSequenced {
            let udp_socket = state.udp_socket.as_ref().map(Arc::clone);
            let udp_remote = state.udp_remote;
            let udp_token = state.udp_token;
            if let (Some(socket), Some(remote), Some(token)) = (udp_socket, udp_remote, udp_token) {
                let frame = DatagramWire {
                    connection_id,
                    token,
                    sequence: state.next_unreliable_sequence,
                    payload,
                };
                state.next_unreliable_sequence = state.next_unreliable_sequence.saturating_add(1);
                let bytes = match encode_datagram(&frame) {
                    Ok(bytes) => bytes,
                    Err(_) => return,
                };
                tokio::spawn(async move {
                    let _ = socket.send_to(&bytes, remote).await;
                });
                return;
            }
        }

        if let Some(reliable_tx) = state.reliable_tx.as_ref() {
            if let Some(telemetry) = state.telemetry.as_ref() {
                let _ = enqueue_reliable_frame(
                    reliable_tx,
                    telemetry,
                    ReliableWireFrame::Payload(PacketWire { lane, payload }),
                );
            }
        }
    }

    async fn disconnect(&self, connection_id: NetworkConnectionId) {
        let mut state = self.state.lock().await;
        if state.connection_id != Some(connection_id) {
            return;
        }
        if let (Some(reliable_tx), Some(telemetry)) =
            (state.reliable_tx.as_ref(), state.telemetry.as_ref())
        {
            let _ = enqueue_reliable_frame(
                reliable_tx,
                telemetry,
                ReliableWireFrame::Disconnect {
                    reason: "client requested disconnect".to_owned(),
                },
            );
        }
        state.connection_id = None;
        state.reliable_tx = None;
        state.telemetry = None;
        state.udp_socket = None;
        state.udp_remote = None;
        state.udp_token = None;
        self.telemetry.remove(connection_id);
    }
}

async fn clear_client_state(
    state: &Arc<tokio::sync::Mutex<ClientState>>,
    telemetry: &NativeTelemetryMap,
) -> Option<NetworkConnectionId> {
    let mut state_guard = state.lock().await;
    let connection_id = state_guard.connection_id;
    state_guard.connection_id = None;
    state_guard.reliable_tx = None;
    state_guard.telemetry = None;
    state_guard.udp_socket = None;
    state_guard.udp_remote = None;
    state_guard.udp_token = None;
    if let Some(connection_id) = connection_id {
        telemetry.remove(connection_id);
    }
    connection_id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn inactive_client_runtime_reports_inactive() {
        let events = Arc::new(Mutex::new(VecDeque::new()));
        let client = ClientRuntime::new(events, NativeTelemetryMap::default());
        assert!(!client.is_active().await);
    }

    #[tokio::test]
    async fn active_client_runtime_reports_active() {
        let events = Arc::new(Mutex::new(VecDeque::new()));
        let client = ClientRuntime::new(events, NativeTelemetryMap::default());
        let (tx, _rx) = mpsc::unbounded_channel();
        {
            let mut state = client.state.lock().await;
            state.reliable_tx = Some(tx);
        }
        assert!(client.is_active().await);
    }

    #[tokio::test]
    async fn native_driver_drops_inactive_client_mode_back_to_idle() {
        let events = Arc::new(Mutex::new(VecDeque::new()));
        let telemetry = NativeTelemetryMap::default();
        let client = ClientRuntime::new(Arc::clone(&events), telemetry.clone());
        let mut driver = NativeDriver {
            events,
            telemetry,
            mode: DriverMode::Client(client),
        };
        driver.refresh_mode().await;
        assert!(matches!(driver.mode, DriverMode::Idle));
    }
}

async fn run_client_udp(
    socket: Arc<UdpSocket>,
    state: Arc<tokio::sync::Mutex<ClientState>>,
    events: Arc<Mutex<VecDeque<NetworkEvent>>>,
) {
    let mut buffer = vec![0_u8; 65_535];
    loop {
        let (len, addr) = match socket.recv_from(&mut buffer).await {
            Ok(value) => value,
            Err(err) => {
                push_event(
                    &events,
                    NetworkEvent::Error {
                        context: "client_udp_recv".to_owned(),
                        message: err.to_string(),
                    },
                );
                break;
            }
        };

        let frame: DatagramWire = match decode_datagram(&buffer[..len]) {
            Ok(frame) => frame,
            Err(err) => {
                push_event(
                    &events,
                    NetworkEvent::Error {
                        context: "client_udp_decode".to_owned(),
                        message: err.to_string(),
                    },
                );
                continue;
            }
        };

        let accepted = {
            let mut state_guard = state.lock().await;
            let expected_id = state_guard.connection_id;
            let expected_token = state_guard.udp_token;
            let expected_remote = state_guard.udp_remote;
            if expected_id != Some(frame.connection_id)
                || expected_token != Some(frame.token)
                || expected_remote != Some(addr)
                || frame.sequence <= state_guard.last_unreliable_sequence
            {
                false
            } else {
                state_guard.last_unreliable_sequence = frame.sequence;
                true
            }
        };

        if accepted {
            push_event(
                &events,
                NetworkEvent::Packet {
                    connection_id: frame.connection_id,
                    lane: NetworkLane::UnreliableSequenced,
                    payload: frame.payload,
                },
            );
        }
    }
}

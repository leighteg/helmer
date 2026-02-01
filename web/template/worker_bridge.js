const WORKER_URL = (() => {
  if (typeof self !== "undefined" && self.location) {
    return new URL("asset_worker.js", self.location.href);
  }
  return new URL("./asset_worker.js", import.meta.url);
})();

const BRIDGE_KEY = "__helmerWorkerBridge";
const GLOBAL_SCOPE =
  typeof globalThis !== "undefined"
    ? globalThis
    : typeof self !== "undefined"
      ? self
      : window;

const bridge =
  GLOBAL_SCOPE[BRIDGE_KEY] ||
  (GLOBAL_SCOPE[BRIDGE_KEY] = {
    wasmExports: null,
    responseHandler: null,
    workers: [],
    pendingResponses: [],
    nextIndex: 0,
    desiredWorkerCount: 0,
    warnedNoWorker: false,
  });

function defaultWorkerCount() {
  const concurrency =
    typeof navigator !== "undefined" && navigator.hardwareConcurrency
      ? navigator.hardwareConcurrency
      : 4;
  return Math.max(1, Math.floor(concurrency) - 1);
}

function ensureWorkers() {
  if (bridge.workers.length) {
    return true;
  }
  if (typeof Worker === "undefined") {
    if (!bridge.warnedNoWorker) {
      console.warn("Web Worker API unavailable; asset workers disabled.");
      bridge.warnedNoWorker = true;
    }
    return false;
  }
  const count =
    bridge.desiredWorkerCount > 0
      ? bridge.desiredWorkerCount
      : defaultWorkerCount();
  for (let i = 0; i < count; i += 1) {
    const worker = new Worker(WORKER_URL, {
      type: "module",
      name: `helmer-asset-worker-${i}`,
    });
    worker.onmessage = (event) => handleWorkerMessage(event.data);
    worker.onerror = (event) => {
      console.error("Asset worker error:", event);
    };
    worker.onmessageerror = (event) => {
      console.error("Asset worker message error:", event);
    };
    bridge.workers.push(worker);
  }
  return bridge.workers.length > 0;
}

function handleWorkerMessage(payload) {
  if (typeof bridge.responseHandler === "function") {
    bridge.responseHandler(payload);
    return;
  }
  if (
    !bridge.wasmExports ||
    typeof bridge.wasmExports.helmer_worker_response !== "function"
  ) {
    bridge.pendingResponses.push(payload);
    return;
  }
  bridge.wasmExports.helmer_worker_response(payload);
}

function flushPendingResponses() {
  if (typeof bridge.responseHandler === "function") {
    if (!bridge.pendingResponses.length) {
      return;
    }
    for (const payload of bridge.pendingResponses) {
      bridge.responseHandler(payload);
    }
    bridge.pendingResponses = [];
    return;
  }
  if (
    !bridge.wasmExports ||
    typeof bridge.wasmExports.helmer_worker_response !== "function"
  ) {
    return;
  }
  if (!bridge.pendingResponses.length) {
    return;
  }
  for (const payload of bridge.pendingResponses) {
    bridge.wasmExports.helmer_worker_response(payload);
  }
  bridge.pendingResponses = [];
}

export function registerHelmerWorkerBridge(wasm) {
  bridge.wasmExports = wasm;
  if (
    wasm &&
    typeof wasm.helmer_worker_response === "function" &&
    wasm.helmer_worker_response.length === 1
  ) {
    bridge.responseHandler = wasm.helmer_worker_response;
  }
  flushPendingResponses();
}

export function helmer_worker_register_callback(callback) {
  if (typeof callback !== "function") {
    return false;
  }
  bridge.responseHandler = callback;
  flushPendingResponses();
  return true;
}

export function helmer_worker_init(workerCount) {
  if (typeof workerCount === "number" && workerCount > 0) {
    bridge.desiredWorkerCount = Math.floor(workerCount);
  }
  return ensureWorkers();
}

export function helmer_worker_enqueue(payload) {
  if (!ensureWorkers()) {
    return false;
  }
  const worker = bridge.workers[bridge.nextIndex++ % bridge.workers.length];
  const bytes = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
  const transfer = new Uint8Array(bytes);
  worker.postMessage(transfer, [transfer.buffer]);
  return true;
}

export function helmer_worker_enqueue_on_worker(workerIndex, payload) {
  if (!ensureWorkers()) {
    return false;
  }
  const indexValue = Number.isFinite(workerIndex)
    ? Math.abs(Math.floor(workerIndex))
    : 0;
  const worker = bridge.workers[indexValue % bridge.workers.length];
  const bytes = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
  const transfer = new Uint8Array(bytes);
  worker.postMessage(transfer, [transfer.buffer]);
  return true;
}

export function helmer_worker_set_opfs_enabled(enabled) {
  if (!ensureWorkers()) {
    return false;
  }
  const value = Boolean(enabled);
  for (const worker of bridge.workers) {
    worker.postMessage({ type: "opfs", enabled: value });
  }
  return true;
}

export function helmer_worker_store_virtual_asset(path, bytes) {
  if (!ensureWorkers()) {
    return false;
  }
  const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const assetPath = String(path || "");
  for (const worker of bridge.workers) {
    const copy = new Uint8Array(data);
    worker.postMessage(
      { type: "virtual-asset", path: assetPath, bytes: copy },
      [copy.buffer],
    );
  }
  return true;
}

export function helmer_worker_release_scene_buffers(sceneId) {
  if (!ensureWorkers()) {
    return false;
  }
  const id = Number(sceneId);
  for (const worker of bridge.workers) {
    worker.postMessage({ type: "release-scene", sceneId: id });
  }
  return true;
}

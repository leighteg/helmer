use hashbrown::HashMap;
use parking_lot::RwLock;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use web_time::Instant;

#[derive(Debug)]
pub struct SystemProfiler {
    enabled: AtomicBool,
    auto_enable_new_systems: AtomicBool,
    entries: RwLock<HashMap<&'static str, Arc<SystemProfileEntry>>>,
}

impl Default for SystemProfiler {
    fn default() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            auto_enable_new_systems: AtomicBool::new(false),
            entries: RwLock::new(HashMap::new()),
        }
    }
}

impl SystemProfiler {
    #[inline]
    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    #[inline]
    pub fn auto_enable_new_systems(&self) -> bool {
        self.auto_enable_new_systems.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn set_auto_enable_new_systems(&self, enabled: bool) {
        self.auto_enable_new_systems
            .store(enabled, Ordering::Relaxed);
    }

    pub fn register_system(&self, name: &'static str) {
        let _ = self.entry_for(name);
    }

    pub fn register_systems(&self, names: &[&'static str]) {
        for &name in names {
            self.register_system(name);
        }
    }

    #[inline]
    pub fn begin_scope(&self, name: &'static str) -> Option<SystemProfileScope> {
        if !self.enabled() {
            return None;
        }
        let entry = self.entry_for(name);
        if !entry.enabled.load(Ordering::Relaxed) {
            return None;
        }
        Some(SystemProfileScope {
            entry,
            start: Instant::now(),
        })
    }

    pub fn set_all_systems_enabled(&self, enabled: bool) {
        self.set_auto_enable_new_systems(enabled);
        let entries = self.entries.read();
        for entry in entries.values() {
            entry.enabled.store(enabled, Ordering::Relaxed);
        }
    }

    pub fn set_system_enabled(&self, name: &str, enabled: bool) -> bool {
        let entries = self.entries.read();
        let Some(entry) = entries.get(name) else {
            return false;
        };
        entry.enabled.store(enabled, Ordering::Relaxed);
        true
    }

    pub fn reset_all(&self) {
        let entries = self.entries.read();
        for entry in entries.values() {
            entry.reset();
        }
    }

    pub fn snapshots(&self) -> Vec<SystemProfileSnapshot> {
        let entries = self.entries.read();
        let mut snapshots = Vec::with_capacity(entries.len());
        for entry in entries.values() {
            let calls = entry.calls.load(Ordering::Relaxed);
            let total_us = entry.total_us.load(Ordering::Relaxed);
            let last_us = entry.last_us.load(Ordering::Relaxed);
            let max_us = entry.max_us.load(Ordering::Relaxed);
            let avg_us = if calls > 0 {
                total_us as f64 / calls as f64
            } else {
                0.0
            };
            snapshots.push(SystemProfileSnapshot {
                name: entry.name,
                enabled: entry.enabled.load(Ordering::Relaxed),
                calls,
                total_us,
                last_us,
                avg_us,
                max_us,
            });
        }
        snapshots
    }

    fn entry_for(&self, name: &'static str) -> Arc<SystemProfileEntry> {
        if let Some(entry) = self.entries.read().get(name).cloned() {
            return entry;
        }

        let mut entries = self.entries.write();
        entries
            .entry(name)
            .or_insert_with(|| {
                Arc::new(SystemProfileEntry::new(
                    name,
                    self.auto_enable_new_systems(),
                ))
            })
            .clone()
    }
}

#[derive(Debug)]
struct SystemProfileEntry {
    name: &'static str,
    enabled: AtomicBool,
    calls: AtomicU64,
    total_us: AtomicU64,
    last_us: AtomicU64,
    max_us: AtomicU64,
}

impl SystemProfileEntry {
    fn new(name: &'static str, enabled: bool) -> Self {
        Self {
            name,
            enabled: AtomicBool::new(enabled),
            calls: AtomicU64::new(0),
            total_us: AtomicU64::new(0),
            last_us: AtomicU64::new(0),
            max_us: AtomicU64::new(0),
        }
    }

    #[inline]
    fn reset(&self) {
        self.calls.store(0, Ordering::Relaxed);
        self.total_us.store(0, Ordering::Relaxed);
        self.last_us.store(0, Ordering::Relaxed);
        self.max_us.store(0, Ordering::Relaxed);
    }

    #[inline]
    fn record_duration(&self, duration_us: u64) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.total_us.fetch_add(duration_us, Ordering::Relaxed);
        self.last_us.store(duration_us, Ordering::Relaxed);

        let mut current_max = self.max_us.load(Ordering::Relaxed);
        while duration_us > current_max {
            match self.max_us.compare_exchange_weak(
                current_max,
                duration_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(next_max) => current_max = next_max,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SystemProfileSnapshot {
    pub name: &'static str,
    pub enabled: bool,
    pub calls: u64,
    pub total_us: u64,
    pub last_us: u64,
    pub avg_us: f64,
    pub max_us: u64,
}

pub struct SystemProfileScope {
    entry: Arc<SystemProfileEntry>,
    start: Instant,
}

impl SystemProfileScope {
    #[inline]
    pub fn finish(self) {}
}

impl Drop for SystemProfileScope {
    fn drop(&mut self) {
        let elapsed_us = self.start.elapsed().as_micros() as u64;
        self.entry.record_duration(elapsed_us);
    }
}

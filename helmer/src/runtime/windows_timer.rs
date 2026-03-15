#[cfg(target_os = "windows")]
mod imp {
    use std::sync::{Mutex, OnceLock};

    #[link(name = "winmm")]
    unsafe extern "system" {
        fn timeBeginPeriod(period: u32) -> u32;
        fn timeEndPeriod(period: u32) -> u32;
    }

    struct TimerState {
        refs: usize,
        period_ms: u32,
    }

    fn timer_state() -> &'static Mutex<TimerState> {
        static STATE: OnceLock<Mutex<TimerState>> = OnceLock::new();
        STATE.get_or_init(|| {
            Mutex::new(TimerState {
                refs: 0,
                period_ms: 1,
            })
        })
    }

    pub struct HighResolutionTimerGuard {
        enabled: bool,
    }

    impl HighResolutionTimerGuard {
        pub fn new(period_ms: u32) -> Self {
            let period_ms = period_ms.max(1);
            let mut guard = Self { enabled: false };
            let Ok(mut state) = timer_state().lock() else {
                tracing::warn!("failed to lock windows timer state");
                return guard;
            };

            if state.refs == 0 {
                let status = unsafe { timeBeginPeriod(period_ms) };
                if status != 0 {
                    tracing::warn!(
                        period_ms,
                        status,
                        "timeBeginPeriod failed; high-resolution timing disabled"
                    );
                    return guard;
                }
                state.period_ms = period_ms;
            }

            state.refs = state.refs.saturating_add(1);
            guard.enabled = true;
            guard
        }
    }

    impl Drop for HighResolutionTimerGuard {
        fn drop(&mut self) {
            if !self.enabled {
                return;
            }
            let Ok(mut state) = timer_state().lock() else {
                return;
            };
            if state.refs == 0 {
                return;
            }
            state.refs -= 1;
            if state.refs == 0 {
                unsafe {
                    let _ = timeEndPeriod(state.period_ms);
                }
            }
        }
    }
}

#[cfg(not(target_os = "windows"))]
mod imp {
    pub struct HighResolutionTimerGuard;

    impl HighResolutionTimerGuard {
        pub fn new(_period_ms: u32) -> Self {
            Self
        }
    }
}

pub use imp::HighResolutionTimerGuard;

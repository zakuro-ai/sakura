//! Tokio multi-thread runtime owned by the crate, hosted in a Rust-managed
//! OS thread so Python's GIL is never blocked by async I/O.

use std::sync::{Arc, OnceLock};
use tokio::runtime::{Builder, Runtime};

/// Lazily-built shared tokio runtime. All async work in sakura-wire runs here.
pub struct WireRuntime {
    rt: Arc<Runtime>,
}

impl WireRuntime {
    pub fn shared() -> &'static WireRuntime {
        static SHARED: OnceLock<WireRuntime> = OnceLock::new();
        SHARED.get_or_init(|| WireRuntime {
            rt: Arc::new(
                Builder::new_multi_thread()
                    .worker_threads(num_cpus_capped(8))
                    .thread_name("sakura-wire")
                    .enable_all()
                    .build()
                    .expect("failed to build tokio runtime"),
            ),
        })
    }

    pub fn handle(&self) -> &tokio::runtime::Handle {
        self.rt.handle()
    }

    pub fn spawn<F>(&self, fut: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.rt.spawn(fut)
    }

    pub fn block_on<F: std::future::Future>(&self, fut: F) -> F::Output {
        self.rt.block_on(fut)
    }
}

fn num_cpus_capped(cap: usize) -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(cap).max(2))
        .unwrap_or(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn shared_runtime_is_singleton() {
        let a = WireRuntime::shared() as *const WireRuntime;
        let b = WireRuntime::shared() as *const WireRuntime;
        assert_eq!(a, b, "WireRuntime::shared must return the same instance");
    }

    #[test]
    fn spawn_runs_to_completion() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c2 = Arc::clone(&counter);
        let handle = WireRuntime::shared().spawn(async move {
            c2.fetch_add(1, Ordering::SeqCst);
            42
        });
        let result = WireRuntime::shared().block_on(handle).expect("join");
        assert_eq!(result, 42);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn block_on_yields_value() {
        let v = WireRuntime::shared().block_on(async { 1 + 1 });
        assert_eq!(v, 2);
    }
}

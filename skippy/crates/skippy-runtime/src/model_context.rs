//! Serialized access to the llama context a model's sessions share.
//!
//! The native runtime gives every session of one model the same
//! `llama_context`: a session is an execution lane (a sequence id) in it, not
//! a context of its own. Native code locks only the lane claim, so two
//! sessions of one model that run on different threads race on the context.
//! Each model owns one [`ModelContextLock`] and every session holds a clone.
//! The native session handle can only be reached through a held lock, so any
//! call into the shared context from Rust is serialized.
//!
//! The lock is reentrant: a session operation can call another session
//! operation of the same model on the same thread.

use std::sync::Arc;

use anyhow::{Result, anyhow};
use parking_lot::{ArcReentrantMutexGuard, RawMutex, RawThreadId, ReentrantMutex};
use skippy_ffi::Session as RawSession;

type Guard = ArcReentrantMutexGuard<RawMutex, RawThreadId, ()>;

/// The execution lock of one model's shared native context.
#[derive(Clone, Default)]
pub(crate) struct ModelContextLock(Arc<ReentrantMutex<()>>);

impl ModelContextLock {
    /// Blocks until this thread owns the model's context.
    pub(crate) fn lock(&self) -> ModelContextGuard {
        ModelContextGuard {
            _guard: self.0.lock_arc(),
        }
    }

    #[cfg(test)]
    fn try_lock(&self) -> Option<ModelContextGuard> {
        self.0
            .try_lock_arc()
            .map(|guard| ModelContextGuard { _guard: guard })
    }

    fn address(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    fn is_held_here(&self) -> bool {
        self.0.is_owned_by_current_thread()
    }
}

/// Proof that this thread owns one model's context until it is dropped.
pub(crate) struct ModelContextGuard {
    _guard: Guard,
}

/// A native session handle that can only be used while its model's context
/// lock is held.
pub(crate) struct NativeSession {
    raw: *mut RawSession,
    context: ModelContextLock,
}

impl NativeSession {
    pub(crate) fn new(raw: *mut RawSession, context: ModelContextLock) -> Self {
        Self { raw, context }
    }

    /// A session without native state, for tests of the Rust bookkeeping.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn detached() -> Self {
        Self::new(std::ptr::null_mut(), ModelContextLock::default())
    }

    /// Takes the model's context lock for one session operation.
    pub(crate) fn lock(&self) -> LockedSession {
        LockedSession {
            raw: self.raw,
            _context: self.context.lock(),
        }
    }

    /// The handle, for a batch operation that already holds `held`.
    pub(crate) fn raw_in(&self, held: &HeldContexts) -> Result<*mut RawSession> {
        if held.covers(&self.context) {
            Ok(self.raw)
        } else {
            Err(anyhow!(
                "skippy session used without its model's context lock"
            ))
        }
    }

    pub(crate) fn is_null(&self) -> bool {
        self.raw.is_null()
    }
}

/// A native session handle whose model context lock this thread holds.
pub(crate) struct LockedSession {
    raw: *mut RawSession,
    _context: ModelContextGuard,
}

impl LockedSession {
    /// The handle. Use it only while this guard is alive: native RAII
    /// guards built from it must be declared after the guard.
    pub(crate) fn raw(&self) -> *mut RawSession {
        self.raw
    }
}

/// The context locks of every model a batch of sessions uses.
pub(crate) struct HeldContexts {
    /// Distinct locks in address order, so that two batches over the same
    /// models cannot each wait on a lock the other holds.
    held: Vec<(usize, ModelContextGuard)>,
}

impl HeldContexts {
    /// Locks the context of every model `sessions` belong to.
    ///
    /// The thread must hold either none of these locks or all of them, as a
    /// batch that retries itself does. Holding only some and waiting for the
    /// rest could deadlock against a thread doing the reverse.
    pub(crate) fn lock<'a>(sessions: impl IntoIterator<Item = &'a NativeSession>) -> Self {
        let mut locks: Vec<&ModelContextLock> = sessions
            .into_iter()
            .map(|session| &session.context)
            .collect();
        locks.sort_by_key(|lock| lock.address());
        locks.dedup_by_key(|lock| lock.address());
        debug_assert!(
            holds_all_or_none(&locks),
            "a batch started while holding only some of its models' context locks"
        );
        Self {
            held: locks
                .into_iter()
                .map(|lock| (lock.address(), lock.lock()))
                .collect(),
        }
    }

    fn covers(&self, lock: &ModelContextLock) -> bool {
        let address = lock.address();
        self.held.iter().any(|(held, _)| *held == address)
    }
}

fn holds_all_or_none(locks: &[&ModelContextLock]) -> bool {
    let held = locks.iter().filter(|lock| lock.is_held_here()).count();
    held == 0 || held == locks.len()
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use super::*;

    fn session_of(context: &ModelContextLock) -> NativeSession {
        NativeSession::new(std::ptr::null_mut(), context.clone())
    }

    fn assert_send<T: Send>() {}

    #[test]
    fn a_session_stays_movable_between_threads() {
        assert_send::<crate::StageSession>();
    }

    #[test]
    fn sessions_of_one_model_exclude_each_other_across_threads() {
        let context = ModelContextLock::default();
        let first = session_of(&context);
        let second = session_of(&context);

        let held = first.lock();
        let second = second.context.clone();
        let contended = thread::spawn(move || second.try_lock().is_none())
            .join()
            .unwrap();
        assert!(
            contended,
            "a second lane ran while the first held the context"
        );
        drop(held);
    }

    #[test]
    fn a_waiting_session_runs_once_the_other_releases_the_context() {
        let context = ModelContextLock::default();
        let first = session_of(&context);
        let second = session_of(&context);
        let (locked, waiting) = mpsc::channel();

        let held = first.lock();
        let second = second.context.clone();
        let waiter = thread::spawn(move || {
            let _held = second.lock();
            locked.send(()).unwrap();
        });
        assert!(
            waiting.recv_timeout(Duration::from_millis(50)).is_err(),
            "a second lane took the context while the first held it"
        );
        drop(held);
        waiting
            .recv_timeout(Duration::from_secs(60))
            .expect("the waiting lane never got the context");
        waiter.join().unwrap();
    }

    #[test]
    fn sessions_of_different_models_do_not_contend() {
        let first = session_of(&ModelContextLock::default());
        let second = session_of(&ModelContextLock::default());

        let _held = first.lock();
        let second = second.context.clone();
        let free = thread::spawn(move || second.try_lock().is_some())
            .join()
            .unwrap();
        assert!(free);
    }

    #[test]
    fn a_session_operation_can_nest_on_one_thread() {
        let context = ModelContextLock::default();
        let first = session_of(&context);
        let second = session_of(&context);

        let _outer = first.lock();
        let _inner = second.lock();
        let _batch = HeldContexts::lock([&first, &second]);
    }

    #[test]
    fn a_batch_can_retry_while_holding_all_its_models() {
        let first = session_of(&ModelContextLock::default());
        let second = session_of(&ModelContextLock::default());

        let _batch = HeldContexts::lock([&first, &second]);
        let retry = HeldContexts::lock([&second, &first]);
        assert!(first.raw_in(&retry).is_ok());
        assert!(second.raw_in(&retry).is_ok());
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "only some of its models' context locks")]
    fn a_batch_rejects_holding_only_some_of_its_models() {
        let first = session_of(&ModelContextLock::default());
        let second = session_of(&ModelContextLock::default());

        let _one = first.lock();
        let _batch = HeldContexts::lock([&first, &second]);
    }

    #[test]
    fn a_batch_holds_each_model_once_and_admits_only_its_sessions() {
        let shared = ModelContextLock::default();
        let first = session_of(&shared);
        let second = session_of(&shared);
        let other = session_of(&ModelContextLock::default());
        let outside = session_of(&ModelContextLock::default());

        let held = HeldContexts::lock([&first, &other, &second]);
        assert_eq!(held.held.len(), 2);
        assert!(first.raw_in(&held).is_ok());
        assert!(second.raw_in(&held).is_ok());
        assert!(other.raw_in(&held).is_ok());
        assert!(outside.raw_in(&held).is_err());

        let blocked = thread::spawn(move || shared.try_lock().is_none())
            .join()
            .unwrap();
        assert!(blocked, "a batch released its model's context early");
    }
}

/// Opt-in: `SKIPPY_CONTEXT_LOCK_MODEL` names a complete causal GGUF.
#[cfg(test)]
mod native_tests {
    use std::thread;

    use anyhow::{Context, Result};

    use crate::{ModelInfo, RuntimeConfig, StageModel, StageSession, TensorRole};

    const LANES: usize = 2;
    const DECODE_STEPS: usize = 16;
    const ROUNDS: usize = 4;

    fn fixture() -> Result<Option<StageModel>> {
        let Some(path) = std::env::var_os("SKIPPY_CONTEXT_LOCK_MODEL").filter(|p| !p.is_empty())
        else {
            return Ok(None);
        };
        let layer_end = ModelInfo::open(&path)?
            .tensors()?
            .into_iter()
            .filter(|tensor| tensor.role == TensorRole::Layer)
            .filter_map(|tensor| tensor.layer_index)
            .max()
            .context("fixture layers")?
            + 1;
        let config = RuntimeConfig {
            layer_end,
            ctx_size: 1024,
            lane_count: u32::try_from(LANES)?,
            n_gpu_layers: 0,
            ..RuntimeConfig::default()
        };
        StageModel::open(path, &config).map(Some)
    }

    /// Distinct prompts per lane, so a lane that reads another's state
    /// produces the other's continuation.
    fn prompt(lane: usize) -> Vec<i32> {
        (0..12)
            .map(|i| i32::try_from(3 + lane * 17 + i * 5).unwrap())
            .collect()
    }

    fn greedy(session: &mut StageSession, prompt: &[i32]) -> Result<Vec<i32>> {
        session.reset()?;
        let (last, prefix) = prompt.split_last().context("empty prompt")?;
        session.prefill_chunked(prefix)?;
        let mut token = session.decode_step(*last)?;
        let mut tokens = vec![token];
        for _ in 1..DECODE_STEPS {
            token = session.decode_step(token)?;
            tokens.push(token);
        }
        Ok(tokens)
    }

    #[test]
    fn lanes_of_one_model_decode_on_separate_threads() -> Result<()> {
        let Some(model) = fixture()? else {
            return Ok(());
        };
        let mut sessions = (0..LANES)
            .map(|_| model.create_session())
            .collect::<Result<Vec<_>>>()?;
        let expected = sessions
            .iter_mut()
            .enumerate()
            .map(|(lane, session)| greedy(session, &prompt(lane)))
            .collect::<Result<Vec<_>>>()?;

        for _ in 0..ROUNDS {
            let decoded = thread::scope(|scope| {
                let workers = sessions
                    .iter_mut()
                    .enumerate()
                    .map(|(lane, session)| scope.spawn(move || greedy(session, &prompt(lane))))
                    .collect::<Vec<_>>();
                workers
                    .into_iter()
                    .map(|worker| worker.join().expect("decode thread panicked"))
                    .collect::<Result<Vec<_>>>()
            })?;
            assert_eq!(
                decoded, expected,
                "concurrent lanes changed each other's output"
            );
        }
        Ok(())
    }
}

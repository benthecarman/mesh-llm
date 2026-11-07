//! Lanes lent out of the active-session map for one batched native call.
//!
//! A batched call needs mutable access to several sessions at once, so it
//! takes them out of [`RuntimeState::sessions`](super::RuntimeState) for the
//! duration of the call. [`CheckedOutLanes`] puts them back when it is
//! dropped, which also happens while a panic unwinds. Panic recovery then
//! finds every lane in the map and returns it to the pool; without that, each
//! panic inside a batch lost its lanes and the runtime eventually rejected
//! every request with "all execution lanes are busy".

use std::collections::BTreeMap;

use anyhow::{Result, anyhow};

use super::RuntimeLaneSession;

/// Sessions taken out of the active-session map, returned to it on drop.
pub(super) struct CheckedOutLanes<'a> {
    sessions: &'a mut BTreeMap<String, RuntimeLaneSession>,
    lanes: Vec<(String, RuntimeLaneSession)>,
}

impl<'a> CheckedOutLanes<'a> {
    /// Takes `session_ids` out of `sessions`, in order. Fails if one is not
    /// active; the sessions already taken go back to the map.
    pub(super) fn take<'id>(
        sessions: &'a mut BTreeMap<String, RuntimeLaneSession>,
        session_ids: impl IntoIterator<Item = &'id str>,
    ) -> Result<Self> {
        let mut checked_out = Self {
            sessions,
            lanes: Vec::new(),
        };
        for session_id in session_ids {
            let lane = checked_out
                .sessions
                .remove(session_id)
                .ok_or_else(|| anyhow!("session {session_id} was not active after admission"))?;
            checked_out.lanes.push((session_id.to_string(), lane));
        }
        Ok(checked_out)
    }

    /// The checked-out lanes, in the order their IDs were given.
    pub(super) fn iter_mut(&mut self) -> impl Iterator<Item = &mut RuntimeLaneSession> {
        self.lanes.iter_mut().map(|(_, lane)| lane)
    }
}

impl Drop for CheckedOutLanes<'_> {
    fn drop(&mut self) {
        for (session_id, lane) in self.lanes.drain(..) {
            self.sessions.insert(session_id, lane);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::thread;

    use super::*;
    use crate::runtime_state::RuntimeState;

    fn runtime_with_active_sessions(session_ids: &[&str]) -> RuntimeState {
        let lane_count = u32::try_from(session_ids.len()).unwrap();
        let mut runtime = RuntimeState::new_modelless_for_test(lane_count);
        for (index, session_id) in session_ids.iter().enumerate() {
            runtime.sessions.insert(
                (*session_id).to_string(),
                RuntimeLaneSession {
                    index,
                    session: skippy_runtime::StageSession::detached_dflash_for_test(),
                    resident_prefix: None,
                },
            );
            runtime.track_session_tokens_for_test(session_id, 1);
        }
        runtime.next_lane_index = session_ids.len();
        runtime
    }

    #[test]
    fn checked_out_lanes_return_to_the_map_on_drop() {
        let mut runtime = runtime_with_active_sessions(&["a", "b"]);
        {
            let mut lanes = CheckedOutLanes::take(&mut runtime.sessions, ["b", "a"]).unwrap();
            let indices = lanes.iter_mut().map(|lane| lane.index).collect::<Vec<_>>();
            assert_eq!(indices, [1, 0], "lanes follow the requested order");
        }
        assert_eq!(runtime.sessions["a"].index, 0);
        assert_eq!(runtime.sessions["b"].index, 1);
    }

    #[test]
    fn failed_checkout_returns_the_lanes_already_taken() {
        let mut runtime = runtime_with_active_sessions(&["a", "b"]);
        let error = CheckedOutLanes::take(&mut runtime.sessions, ["a", "missing", "b"])
            .err()
            .expect("an inactive session fails the checkout");
        assert!(error.to_string().contains("missing"), "{error:#}");
        assert_eq!(runtime.sessions.len(), 2);
    }

    /// Panic recovery resets every lane it finds in the session map (see
    /// `panic_recovery`); a lane is lost only if it is missing from the map.
    /// Recovery itself resets lanes through the native runtime, which a test
    /// build may link statically or leave unloaded, so this checks the state
    /// recovery starts from.
    #[test]
    fn panic_during_a_batch_leaves_its_lanes_for_recovery() {
        let runtime = Arc::new(Mutex::new(runtime_with_active_sessions(&["a", "b"])));
        let panicking = Arc::clone(&runtime);
        let _ = thread::spawn(move || {
            let mut runtime = panicking.lock().unwrap();
            let mut lanes = CheckedOutLanes::take(&mut runtime.sessions, ["a", "b"]).unwrap();
            let _ = lanes.iter_mut().count();
            panic!("batched native call panicked");
        })
        .join();

        let Err(poisoned) = runtime.lock() else {
            panic!("precondition: the panic poisoned the runtime");
        };
        let runtime = poisoned.into_inner();
        let indices = runtime
            .sessions
            .iter()
            .map(|(session_id, lane)| (session_id.as_str(), lane.index))
            .collect::<Vec<_>>();
        assert_eq!(indices, [("a", 0), ("b", 1)]);
    }
}

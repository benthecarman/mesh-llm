//! Decodes run at the position stage 0 names for them.
//!
//! A stage decodes, and verifies a window, from its session's own position,
//! so the position a decode message carries is a check, not an input: a
//! session that is behind, or one this stage does not know, must fail the
//! decode rather than run it at the wrong position.

use anyhow::{Context, Result};
use skippy_protocol::binary::{StageWireMessage, WireMessageKind};

use crate::runtime_state::RuntimeState;

/// The position a decode or verify window must run at, or `None` for a
/// message that is not a positioned decode.
pub(in crate::binary_transport) fn decode_position(
    message: &StageWireMessage,
) -> Result<Option<u64>> {
    if !matches!(
        message.kind,
        WireMessageKind::DecodeEmbd
            | WireMessageKind::DecodeReadout
            | WireMessageKind::DecodeLightCtx
            | WireMessageKind::VerifyWindow
    ) {
        return Ok(None);
    }
    message
        .authoritative_session_position()
        .with_context(|| {
            format!(
                "{:?} decode has negative position {}",
                message.kind, message.pos_start
            )
        })
        .map(Some)
}

/// Fails a decode whose session does not stand at the decode's position.
pub(in crate::binary_transport) fn ensure_decode_position(
    runtime: &RuntimeState,
    session_id: &str,
    message: &StageWireMessage,
) -> Result<()> {
    let Some(position) = decode_position(message)? else {
        return Ok(());
    };
    runtime
        .ensure_session_at_position(session_id, position)
        .with_context(|| format!("reject {:?} decode", message.kind))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary_transport::{BinaryStageExecutionOptions, run_binary_stage_message};
    use skippy_protocol::binary::StageStateHeader;

    const POSITIONED_DECODES: [WireMessageKind; 4] = [
        WireMessageKind::DecodeEmbd,
        WireMessageKind::DecodeReadout,
        WireMessageKind::DecodeLightCtx,
        WireMessageKind::VerifyWindow,
    ];

    fn message(kind: WireMessageKind, pos_start: i32) -> StageWireMessage {
        let mut state = StageStateHeader::new(kind);
        state.current_token = 7;
        StageWireMessage {
            kind,
            pos_start,
            token_count: 1,
            state,
            request_id: 1,
            session_id: 1,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: vec![7],
            positions: Vec::new(),
            activation: Vec::new(),
            readout_tokens: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }

    fn run(runtime: &mut RuntimeState, message: &StageWireMessage) -> Result<()> {
        run_binary_stage_message(
            runtime,
            "s",
            message,
            &[7],
            None,
            BinaryStageExecutionOptions::new(false, 0, false),
        )
        .map(|_| ())
    }

    #[test]
    fn a_decode_for_an_unknown_session_does_not_start_one() {
        let mut runtime = RuntimeState::new_modelless_for_test(1);
        for kind in POSITIONED_DECODES {
            let error = run(&mut runtime, &message(kind, 5)).unwrap_err();
            assert!(
                format!("{error:#}").contains("is not active"),
                "{kind:?}: {error:#}"
            );
        }
        assert_eq!(runtime.session_token_count("s"), None);
        assert_eq!(runtime.session_stats().active_sessions, 0);
    }

    #[test]
    fn a_decode_for_a_session_behind_its_position_is_rejected() {
        let mut runtime = RuntimeState::new_modelless_for_test(1);
        runtime.track_session_tokens_for_test("s", 3);
        for kind in POSITIONED_DECODES {
            let error = run(&mut runtime, &message(kind, 5)).unwrap_err();
            assert!(
                format!("{error:#}").contains("is at position 3, but the decode is at position 5"),
                "{kind:?}: {error:#}"
            );
        }
        assert_eq!(runtime.session_token_count("s"), Some(3));
    }

    #[test]
    fn a_decode_at_the_session_position_is_admitted() {
        let mut runtime = RuntimeState::new_modelless_for_test(1);
        runtime.track_session_tokens_for_test("s", 5);
        for kind in POSITIONED_DECODES {
            assert!(ensure_decode_position(&runtime, "s", &message(kind, 5)).is_ok());
        }
    }

    #[test]
    fn only_a_decode_at_position_zero_may_start_a_session() {
        // The first decode of a one-token prompt reaches a stage that has
        // not prefilled anything for the session.
        let runtime = RuntimeState::new_modelless_for_test(1);
        assert!(
            ensure_decode_position(&runtime, "s", &message(WireMessageKind::DecodeEmbd, 0)).is_ok()
        );
    }

    #[test]
    fn a_decode_with_a_negative_position_is_rejected() {
        let runtime = RuntimeState::new_modelless_for_test(1);
        for kind in POSITIONED_DECODES {
            assert!(ensure_decode_position(&runtime, "s", &message(kind, -1)).is_err());
        }
    }

    #[test]
    fn messages_other_than_positioned_decodes_are_not_checked() {
        let runtime = RuntimeState::new_modelless_for_test(1);
        for kind in [
            WireMessageKind::PrefillEmbd,
            WireMessageKind::PrefillReadout,
            WireMessageKind::DecodeReplayEmbd,
        ] {
            assert_eq!(decode_position(&message(kind, 5)).unwrap(), None);
            assert!(ensure_decode_position(&runtime, "s", &message(kind, 5)).is_ok());
        }
    }
}

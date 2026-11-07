//! Planning-input checks shared by the automatic and locked planners.
//!
//! Every per-layer vector must be absent or aligned to the model's layers: a
//! misaligned one would silently fall back to an even share or to zero and
//! misprice every stage.

use super::{TopologyPlanError, TopologyPlanningInput};

pub(super) fn validate_input(input: &TopologyPlanningInput) -> Result<(), TopologyPlanError> {
    if input.native_context_length == 0 {
        return Err(TopologyPlanError::MissingNativeContext);
    }
    if input.layer_count == 0 {
        return Err(TopologyPlanError::MissingLayers);
    }
    if input.model_weight_bytes == 0 {
        return Err(TopologyPlanError::MissingModelWeights);
    }
    validate_layer_vector(
        "layer_weight_bytes",
        &input.layer_weight_bytes,
        input.layer_count,
    )?;
    validate_layer_vector(
        "recurrent_bytes_per_sequence_by_layer",
        &input.recurrent_bytes_per_sequence_by_layer,
        input.layer_count,
    )?;
    let recurrent_only = input.recurrent_bytes_per_sequence_by_layer.len()
        == input.layer_count as usize
        && input
            .recurrent_bytes_per_sequence_by_layer
            .iter()
            .all(|bytes| *bytes > 0);
    if input.kv_bytes_per_token == 0 && !recurrent_only {
        return Err(TopologyPlanError::MissingKvBytesPerToken);
    }
    if input.nodes.is_empty() {
        return Err(TopologyPlanError::MissingNodes);
    }
    Ok(())
}

/// A per-layer vector is either absent (empty) or has one entry per layer.
/// A misaligned vector would otherwise be dropped for an even-share or zero
/// fallback and misprice every stage.
fn validate_layer_vector(
    field: &'static str,
    values: &[u64],
    layer_count: u32,
) -> Result<(), TopologyPlanError> {
    if values.is_empty() || values.len() == layer_count as usize {
        return Ok(());
    }
    Err(TopologyPlanError::LayerVectorLengthMismatch {
        field,
        found: values.len(),
        layer_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{TopologyNode, plan_locked_topology, plan_topology};

    const GIB: u64 = 1024 * 1024 * 1024;

    fn input() -> TopologyPlanningInput {
        TopologyPlanningInput {
            native_context_length: 4096,
            layer_count: 4,
            model_weight_bytes: 4 * GIB,
            layer_weight_bytes: Vec::new(),
            kv_bytes_per_token: 1024,
            final_layer_kv_bytes_per_token: 0,
            recurrent_bytes_per_sequence_by_layer: Vec::new(),
            reserved_sequence_ids: 0,
            minimum_nodes: 1,
            nodes: vec![TopologyNode {
                node_id: "node".to_string(),
                detected_vram_bytes: 64 * GIB,
                max_vram_bytes: None,
                runtime_headroom_bytes: 0,
                stage_transfer_latency_ms: None,
                decode_bytes_per_second: None,
            }],
            context_length_override: None,
            parallel_lanes_override: None,
            target_decode_tpot_ms: None,
            auto_balance: false,
            placement_objective: Default::default(),
        }
    }

    #[test]
    fn absent_and_aligned_layer_vectors_are_accepted() {
        let mut aligned = input();
        aligned.layer_weight_bytes = vec![GIB; 4];
        aligned.recurrent_bytes_per_sequence_by_layer = vec![0, 1, 0, 1];
        assert_eq!(validate_input(&input()), Ok(()));
        assert_eq!(validate_input(&aligned), Ok(()));
    }

    #[test]
    fn misaligned_layer_weights_are_rejected() {
        let mut request = input();
        request.layer_weight_bytes = vec![GIB; 5];
        assert_eq!(
            plan_topology(&request),
            Err(TopologyPlanError::LayerVectorLengthMismatch {
                field: "layer_weight_bytes",
                found: 5,
                layer_count: 4,
            })
        );
    }

    #[test]
    fn misaligned_recurrent_bytes_are_rejected_by_every_planner() {
        let mut request = input();
        request.recurrent_bytes_per_sequence_by_layer = vec![GIB; 3];
        let expected = Err(TopologyPlanError::LayerVectorLengthMismatch {
            field: "recurrent_bytes_per_sequence_by_layer",
            found: 3,
            layer_count: 4,
        });
        assert_eq!(plan_topology(&request), expected);
        let locked = [crate::topology::LockedTopologyStage {
            node_id: "node".to_string(),
            layer_start: 0,
            layer_end: 4,
        }];
        assert_eq!(plan_locked_topology(&request, &locked), expected);
    }

    #[test]
    fn throughput_estimates_reject_invalid_inputs_and_stage_ranges() {
        let mut request = input();
        request.nodes[0].decode_bytes_per_second = Some(GIB);
        let plan = plan_topology(&request).unwrap();
        assert!(crate::topology::estimate_plan_throughput(&request, &plan).is_some());

        let mut outside = plan.clone();
        outside.stages[0].layer_end = 9;
        assert_eq!(
            crate::topology::estimate_plan_throughput(&request, &outside),
            None
        );

        let mut misaligned = request.clone();
        misaligned.layer_weight_bytes = vec![GIB; 2];
        assert_eq!(
            crate::topology::estimate_plan_throughput(&misaligned, &plan),
            None
        );
        assert_eq!(
            crate::topology::rebalance_topology(&misaligned, &plan),
            None
        );
    }
}

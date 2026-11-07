use crate::edge_order::StageEdgeSignal;
use crate::validation::{PlanError, validate_request};
use crate::{
    BoundaryDecision, BoundaryPlan, DiagnosticSeverity, ExactStateMobility, FamilyCapabilityRecord,
    LayerSpec, MigrationPolicy, NodePlacementSignal, NodeSpec, PlanDiagnostic, PlanReasonCode,
    PlannerPolicy, SidebandKind, SplitConstraintKind, StagePlan, StageRole, StateAffinity,
    TopologyPlan, TopologyPlanRequest, artifact_diagnostics, edge_order,
};

pub fn plan_even_contiguous(request: &TopologyPlanRequest) -> Result<TopologyPlan, PlanError> {
    validate_request(request)?;

    let stage_count = request.nodes.len().min(request.layers.len());
    let base = request.layers.len() / stage_count;
    let remainder = request.layers.len() % stage_count;
    let mut next_layer = 0usize;
    let mut ranges = Vec::with_capacity(stage_count);

    for stage_index in 0..stage_count {
        let layer_count = base + usize::from(stage_index < remainder);
        ranges.push((next_layer, next_layer + layer_count));
        next_layer += layer_count;
    }

    plan_ranges(request, &ranges)
}

pub fn plan_weighted_contiguous(request: &TopologyPlanRequest) -> Result<TopologyPlan, PlanError> {
    plan_weighted_contiguous_with_signals(request, &[])
}

fn plan_weighted_contiguous_with_signals(
    request: &TopologyPlanRequest,
    placement_signals: &[NodePlacementSignal],
) -> Result<TopologyPlan, PlanError> {
    validate_request(request)?;

    let stage_count = request.nodes.len().min(request.layers.len());
    let nodes = &request.nodes[..stage_count];
    let total_weight: u64 = nodes.iter().map(|node| node.vram_bytes).sum();
    if total_weight == 0 {
        let base = request.layers.len() / stage_count;
        let remainder = request.layers.len() % stage_count;
        let mut next_layer = 0usize;
        let mut ranges = Vec::with_capacity(stage_count);

        for stage_index in 0..stage_count {
            let layer_count = base + usize::from(stage_index < remainder);
            ranges.push((next_layer, next_layer + layer_count));
            next_layer += layer_count;
        }

        return plan_ranges_with_signals(request, &ranges, placement_signals);
    }

    let mut ranges = Vec::with_capacity(stage_count);
    let mut layer_start = 0usize;
    for (stage_index, node) in nodes.iter().enumerate() {
        let remaining_stages = stage_count - stage_index;
        let remaining_layers = request.layers.len() - layer_start;
        let mut span = if remaining_stages == 1 {
            remaining_layers
        } else {
            (((request.layers.len() as u128) * (node.vram_bytes as u128)) / (total_weight as u128))
                .try_into()
                .unwrap_or(usize::MAX)
        };
        span = span.max(1).min(remaining_layers - (remaining_stages - 1));
        let layer_end = layer_start + span;
        ranges.push((layer_start, layer_end));
        layer_start = layer_end;
    }

    plan_ranges_with_signals(request, &ranges, placement_signals)
}

pub fn plan_package_aware_contiguous(
    request: &TopologyPlanRequest,
) -> Result<TopologyPlan, PlanError> {
    plan_package_aware_contiguous_with_signals(request, &[])
}

pub fn plan_package_aware_contiguous_with_signals(
    request: &TopologyPlanRequest,
    placement_signals: &[NodePlacementSignal],
) -> Result<TopologyPlan, PlanError> {
    plan_package_aware_contiguous_with_transport(request, placement_signals, &[])
}

pub fn plan_package_aware_contiguous_with_transport(
    request: &TopologyPlanRequest,
    placement_signals: &[NodePlacementSignal],
    edge_signals: &[StageEdgeSignal],
) -> Result<TopologyPlan, PlanError> {
    validate_request(request)?;

    if !request.nodes.iter().any(|node| node.cached_slice_bytes > 0)
        && !placement_signals.iter().any(has_package_aware_signal)
        && edge_signals.is_empty()
    {
        return plan_weighted_contiguous(request);
    }

    let mut nodes = request
        .nodes
        .iter()
        .cloned()
        .enumerate()
        .collect::<Vec<_>>();
    nodes.sort_by(|(left_index, left), (right_index, right)| {
        let left_signal = placement_signal_for(placement_signals, &left.node_id);
        let right_signal = placement_signal_for(placement_signals, &right.node_id);
        node_package_score(right, right_signal)
            .cmp(&node_package_score(left, left_signal))
            .then_with(|| left_index.cmp(right_index))
            .then_with(|| left.node_id.cmp(&right.node_id))
    });
    let nodes = nodes
        .into_iter()
        .map(|(_, node)| node)
        .enumerate()
        .collect::<Vec<_>>();
    let nodes = edge_order::order_pipeline_nodes(nodes, placement_signals, edge_signals);

    let sorted_request = TopologyPlanRequest {
        topology_id: request.topology_id.clone(),
        model_id: request.model_id.clone(),
        layers: request.layers.clone(),
        nodes: nodes.into_iter().map(|(_, node)| node).collect(),
        family: request.family.clone(),
        policy: request.policy,
    };
    let mut plan = plan_weighted_contiguous_with_signals(&sorted_request, placement_signals)?;
    edge_order::append_edge_diagnostics(&mut plan, edge_signals);
    Ok(plan)
}

pub fn plan_contiguous_with_splits(
    request: &TopologyPlanRequest,
    splits: &[u32],
) -> Result<TopologyPlan, PlanError> {
    validate_request(request)?;

    let layer_start = request
        .layers
        .first()
        .expect("validated non-empty layers")
        .index;
    let layer_end = request
        .layers
        .last()
        .expect("validated non-empty layers")
        .index
        + 1;
    let mut previous = layer_start;
    let mut boundaries = Vec::with_capacity(splits.len() + 2);
    boundaries.push(layer_start);
    for &boundary in splits {
        if boundary <= layer_start || boundary >= layer_end {
            return Err(PlanError::InvalidSplitBoundary {
                boundary,
                layer_start,
                layer_end,
            });
        }
        if boundary <= previous {
            return Err(PlanError::NonAscendingSplitBoundary { previous, boundary });
        }
        boundaries.push(boundary);
        previous = boundary;
    }
    boundaries.push(layer_end);

    let stage_count = boundaries.len() - 1;
    if request.nodes.len() < stage_count {
        return Err(PlanError::NotEnoughNodesForSplits {
            stages: stage_count,
            nodes: request.nodes.len(),
        });
    }

    let first_layer = request.layers[0].index;
    let ranges = boundaries
        .windows(2)
        .map(|window| {
            (
                (window[0] - first_layer) as usize,
                (window[1] - first_layer) as usize,
            )
        })
        .collect::<Vec<_>>();

    plan_ranges(request, &ranges)
}

/// Selects the accepted contiguous split set nearest to an even partition.
///
/// This is the common product policy for callers that need a deterministic
/// balanced topology but cannot assume that arithmetic midpoint or thirds are
/// legal for the model family. Every candidate boundary is evaluated through
/// the same capability rules used by explicit product topology plans.
pub fn plan_balanced_accepted_contiguous(
    request: &TopologyPlanRequest,
    stage_count: usize,
) -> Result<TopologyPlan, PlanError> {
    validate_request(request)?;
    if stage_count < 2 {
        return Err(PlanError::InvalidStageCount {
            stages: stage_count,
        });
    }
    if request.nodes.len() < stage_count {
        return Err(PlanError::NotEnoughNodesForSplits {
            stages: stage_count,
            nodes: request.nodes.len(),
        });
    }
    if request.layers.len() < stage_count {
        return Err(PlanError::NoAcceptedBalancedSplit {
            stages: stage_count,
        });
    }

    let layer_start = request.layers[0].index;
    let layer_end = request.layers.last().expect("validated layers").index + 1;
    let accepted = (layer_start + 1..layer_end)
        .filter(|boundary| {
            plan_contiguous_with_splits(request, &[*boundary]).is_ok_and(|plan| {
                plan.boundaries
                    .iter()
                    .all(|item| item.decision == BoundaryDecision::Accepted)
            })
        })
        .collect::<Vec<_>>();

    let Some(splits) = choose_balanced_splits(&accepted, layer_start, layer_end, stage_count)
    else {
        return Err(PlanError::NoAcceptedBalancedSplit {
            stages: stage_count,
        });
    };
    let plan = plan_contiguous_with_splits(request, &splits)?;
    if plan
        .boundaries
        .iter()
        .any(|boundary| boundary.decision != BoundaryDecision::Accepted)
    {
        return Err(PlanError::NoAcceptedBalancedSplit {
            stages: stage_count,
        });
    }
    Ok(plan)
}

/// Chooses `stage_count - 1` ascending boundaries from `accepted` that are
/// nearest to an even partition of `layer_start..layer_end`.
///
/// The score is the summed distance of each boundary from its even-partition
/// ideal; ties go to the lexicographically smallest boundary set. Because the
/// score is a sum of independent per-position terms, a suffix dynamic program
/// finds the optimum in `O(split_count * accepted.len())` instead of
/// enumerating every combination of accepted boundaries.
///
/// `accepted` must be strictly ascending and lie inside the layer range.
/// Returns `None` when fewer than two stages are requested or `accepted` has
/// too few boundaries.
fn choose_balanced_splits(
    accepted: &[u32],
    layer_start: u32,
    layer_end: u32,
    stage_count: usize,
) -> Option<Vec<u32>> {
    let split_count = stage_count.checked_sub(1)?;
    if split_count == 0 || accepted.len() < split_count {
        return None;
    }
    let cost = BalancedSplitCost {
        layer_start,
        span: u128::from(layer_end - layer_start),
        stage_count: stage_count as u128,
    };
    let best_from = balanced_suffix_costs(accepted, split_count, &cost);
    reconstruct_balanced_splits(accepted, &best_from, &cost)
}

struct BalancedSplitCost {
    layer_start: u32,
    span: u128,
    stage_count: u128,
}

impl BalancedSplitCost {
    /// Distance of `boundary`, used as split `position`, from its even ideal,
    /// scaled by `stage_count` to stay in integers.
    fn at(&self, position: usize, boundary: u32) -> u128 {
        let actual = u128::from(boundary - self.layer_start) * self.stage_count;
        let ideal = self.span * (position as u128 + 1);
        actual.abs_diff(ideal)
    }
}

/// `best_from[position][index]` is the minimum cost of splits
/// `position..split_count` when split `position` is `accepted[index]`, or
/// `None` when too few accepted boundaries follow `index`.
fn balanced_suffix_costs(
    accepted: &[u32],
    split_count: usize,
    cost: &BalancedSplitCost,
) -> Vec<Vec<Option<u128>>> {
    let mut best_from = vec![vec![None; accepted.len()]; split_count];
    for position in (0..split_count).rev() {
        let is_last = position + 1 == split_count;
        // Minimum of the next row over indexes strictly after `index`.
        let mut later_min: Option<u128> = None;
        for index in (0..accepted.len()).rev() {
            let tail = if is_last { Some(0) } else { later_min };
            best_from[position][index] = tail.map(|tail| tail + cost.at(position, accepted[index]));
            if !is_last {
                later_min = later_min
                    .into_iter()
                    .chain(best_from[position + 1][index])
                    .min();
            }
        }
    }
    best_from
}

/// Walks the suffix costs front to back, taking the smallest boundary that
/// still completes an optimal split set. That yields the lexicographically
/// smallest optimum.
fn reconstruct_balanced_splits(
    accepted: &[u32],
    best_from: &[Vec<Option<u128>>],
    cost: &BalancedSplitCost,
) -> Option<Vec<u32>> {
    let mut remaining = best_from.first()?.iter().flatten().min().copied()?;
    let mut splits = Vec::with_capacity(best_from.len());
    let mut start = 0;
    for (position, row) in best_from.iter().enumerate() {
        let index = (start..accepted.len()).find(|&index| row[index] == Some(remaining))?;
        splits.push(accepted[index]);
        remaining -= cost.at(position, accepted[index]);
        start = index + 1;
    }
    Some(splits)
}

fn plan_ranges(
    request: &TopologyPlanRequest,
    ranges: &[(usize, usize)],
) -> Result<TopologyPlan, PlanError> {
    plan_ranges_with_signals(request, ranges, &[])
}

fn plan_ranges_with_signals(
    request: &TopologyPlanRequest,
    ranges: &[(usize, usize)],
    placement_signals: &[NodePlacementSignal],
) -> Result<TopologyPlan, PlanError> {
    let mut stages = Vec::with_capacity(ranges.len());

    for (stage_index, &(start, end)) in ranges.iter().enumerate() {
        let layers = &request.layers[start..end];
        let layer_start = layers.first().expect("validated non-empty range").index;
        let layer_end = layers.last().expect("validated non-empty range").index + 1;
        let state_affinity = classify_layers_with_family(layers, request.family.as_ref());
        let migration_policy = migration_policy(state_affinity, request.policy);
        let parameter_bytes = layers.iter().map(|layer| layer.parameter_bytes).sum();
        let node = &request.nodes[stage_index];
        let node_id = node.node_id.clone();
        let placement_signal = placement_signal_for(placement_signals, &node_id);
        let mut reason_codes =
            stage_reason_codes(state_affinity, migration_policy, request.family.as_ref());
        reason_codes.extend(node_reason_codes(node, placement_signal));

        stages.push(StagePlan {
            stage_id: format!("stage-{stage_index}"),
            stage_index: stage_index as u32,
            node_id,
            roles: stage_roles(stage_index, ranges.len()),
            layer_start,
            layer_end,
            layer_count: (end - start) as u32,
            parameter_bytes,
            state_affinity,
            migration_policy,
            reason_codes,
            cached_slice_bytes: placement_signal
                .map(|signal| signal.cached_slice_bytes.max(node.cached_slice_bytes))
                .unwrap_or(node.cached_slice_bytes),
            missing_artifact_bytes: placement_signal
                .map(|signal| signal.missing_artifact_bytes)
                .unwrap_or_default(),
            rtt_ms: placement_signal.and_then(|signal| signal.rtt_ms),
        });
    }

    let boundaries = boundaries_for(&stages, request.family.as_ref());
    let diagnostics = diagnostics_for(
        &stages,
        &boundaries,
        placement_signals,
        request.family.as_ref(),
        request.policy,
    );

    Ok(TopologyPlan {
        topology_id: request.topology_id.clone(),
        model_id: request.model_id.clone(),
        family_id: request
            .family
            .as_ref()
            .map(|family| family.family_id.clone()),
        stages,
        boundaries,
        diagnostics,
    })
}

fn placement_signal_for<'a>(
    placement_signals: &'a [NodePlacementSignal],
    node_id: &str,
) -> Option<&'a NodePlacementSignal> {
    placement_signals
        .iter()
        .find(|signal| signal.node_id == node_id)
}

fn has_package_aware_signal(signal: &NodePlacementSignal) -> bool {
    signal.cached_slice_bytes > 0
        || signal.missing_artifact_bytes > 0
        || signal.rtt_ms.is_some()
        || signal.artifact_transfer_supported
        || signal.availability_score > 0
}

fn node_package_score(node: &NodeSpec, signal: Option<&NodePlacementSignal>) -> i128 {
    let mut score = i128::from(node.vram_bytes);
    let cached_slice_bytes = signal
        .map(|signal| signal.cached_slice_bytes.max(node.cached_slice_bytes))
        .unwrap_or(node.cached_slice_bytes);
    score += i128::from(cached_slice_bytes).saturating_mul(2);
    if let Some(signal) = signal {
        score -= i128::from(signal.missing_artifact_bytes).saturating_mul(4);
        if signal.missing_artifact_bytes > 0 && !signal.artifact_transfer_supported {
            score -= i128::from(signal.missing_artifact_bytes).saturating_mul(4);
        }
        if let Some(rtt_ms) = signal.rtt_ms {
            score -= i128::from(rtt_ms).saturating_mul(16 * 1024 * 1024);
        }
        score += i128::from(signal.availability_score).saturating_mul(1024 * 1024);
    }
    score
}

fn node_reason_codes(node: &NodeSpec, signal: Option<&NodePlacementSignal>) -> Vec<PlanReasonCode> {
    let mut codes = Vec::new();
    let cached_slice_bytes = signal
        .map(|signal| signal.cached_slice_bytes.max(node.cached_slice_bytes))
        .unwrap_or(node.cached_slice_bytes);
    if cached_slice_bytes > 0 {
        codes.push(PlanReasonCode::CacheLocalityPreferred);
    }
    if let Some(signal) = signal {
        if signal.missing_artifact_bytes > 0 {
            codes.push(PlanReasonCode::ArtifactTransferPenalty);
        }
        if signal.rtt_ms.is_some_and(|rtt| rtt > 0) {
            codes.push(PlanReasonCode::NetworkPipelineCost);
        }
        if signal.availability_score > 0 {
            codes.push(PlanReasonCode::PeerAvailabilityPreferred);
        }
    }
    codes
}

pub fn classify_layers(layers: &[LayerSpec]) -> StateAffinity {
    classify_layers_with_family(layers, None)
}

fn classify_layers_with_family(
    layers: &[LayerSpec],
    family: Option<&FamilyCapabilityRecord>,
) -> StateAffinity {
    let has_attention = layers.iter().any(|layer| layer.attention);
    let has_recurrent = layers.iter().any(|layer| {
        layer.recurrent
            || family.is_some_and(|family| {
                family
                    .recurrent_ranges
                    .iter()
                    .any(|range| range.contains_layer(layer.index))
            })
    });

    match (has_attention, has_recurrent) {
        (false, false) => StateAffinity::Stateless,
        (true, false) => StateAffinity::AttentionKv,
        (false, true) => StateAffinity::Recurrent,
        (true, true) => StateAffinity::Mixed,
    }
}

fn migration_policy(affinity: StateAffinity, policy: PlannerPolicy) -> MigrationPolicy {
    match affinity {
        StateAffinity::Stateless => MigrationPolicy::FreelyMovable,
        StateAffinity::AttentionKv => MigrationPolicy::CostedKv,
        StateAffinity::Recurrent | StateAffinity::Mixed => {
            if policy.allow_recurrent_state_transfer {
                MigrationPolicy::RecurrentStateTransferAllowed
            } else {
                MigrationPolicy::StickyRecurrentOwner
            }
        }
    }
}

fn stage_reason_codes(
    affinity: StateAffinity,
    migration_policy: MigrationPolicy,
    family: Option<&FamilyCapabilityRecord>,
) -> Vec<PlanReasonCode> {
    let mut codes = Vec::new();
    match migration_policy {
        MigrationPolicy::FreelyMovable => {}
        MigrationPolicy::CostedKv => codes.push(PlanReasonCode::AttentionKvCosted),
        MigrationPolicy::StickyRecurrentOwner => codes.push(PlanReasonCode::RecurrentOwnerSticky),
        MigrationPolicy::RecurrentStateTransferAllowed => {
            codes.push(PlanReasonCode::RecurrentStateTransferAllowed)
        }
    }
    if matches!(affinity, StateAffinity::Recurrent | StateAffinity::Mixed)
        && !codes.contains(&PlanReasonCode::RecurrentOwnerSticky)
        && !codes.contains(&PlanReasonCode::RecurrentStateTransferAllowed)
    {
        codes.push(PlanReasonCode::RecurrentOwnerSticky);
    }
    if let Some(family) = family {
        match family.exact_state_mobility {
            ExactStateMobility::Accepted => codes.push(PlanReasonCode::ExactStateMobilityAccepted),
            ExactStateMobility::RejectedTooLarge => {
                codes.push(PlanReasonCode::ExactStateMobilityRejected)
            }
            ExactStateMobility::Untested => {}
        }
    }
    codes
}

fn stage_roles(stage_index: usize, stage_count: usize) -> Vec<StageRole> {
    let mut roles = Vec::new();
    if stage_index == 0 {
        roles.push(StageRole::Driver);
        roles.push(StageRole::Embedding);
    }
    if stage_index + 1 == stage_count {
        roles.push(StageRole::Readout);
    } else if stage_index > 0 {
        roles.push(StageRole::Intermediate);
    }
    roles
}

fn boundaries_for(
    stages: &[StagePlan],
    family: Option<&FamilyCapabilityRecord>,
) -> Vec<BoundaryPlan> {
    stages
        .windows(2)
        .map(|window| {
            let producer = &window[0];
            let consumer = &window[1];
            let layer_boundary = producer.layer_end;
            let mut decision = BoundaryDecision::Accepted;
            let mut reason_codes = vec![PlanReasonCode::ActivationOnlyBoundary];
            let mut messages = vec![format!(
                "activation boundary after layer {}; send activation frame from {} to {}",
                layer_boundary, producer.stage_id, consumer.stage_id
            )];

            if matches!(
                producer.migration_policy,
                MigrationPolicy::StickyRecurrentOwner
            ) || matches!(
                consumer.migration_policy,
                MigrationPolicy::StickyRecurrentOwner
            ) {
                reason_codes.push(PlanReasonCode::RecurrentOwnerSticky);
                messages.push(
                    "recurrent state remains with the owning stage; only activation crosses this boundary"
                        .to_string(),
                );
            }

            let (raw_activation_bytes_per_token, wire_payload_bytes_per_token) =
                if let Some(family) = family {
                    apply_family_boundary_rules(
                        family,
                        layer_boundary,
                        &mut decision,
                        &mut reason_codes,
                        &mut messages,
                    );
                    let payload_multiplier =
                        activation_payload_multiplier_for_boundary(family, layer_boundary);
                    let raw = u64::from(family.activation_width) * 4 * payload_multiplier;
                    let wire = wire_payload_bytes_per_token(family.activation_width)
                        * payload_multiplier;
                    (raw, wire)
                } else {
                    (0, 0)
                };

            BoundaryPlan {
                producer_stage_index: producer.stage_index,
                consumer_stage_index: consumer.stage_index,
                layer_boundary,
                decision,
                raw_activation_bytes_per_token,
                wire_payload_bytes_per_token,
                reason_codes,
                messages,
            }
        })
        .collect()
}

fn apply_family_boundary_rules(
    family: &FamilyCapabilityRecord,
    layer_boundary: u32,
    decision: &mut BoundaryDecision,
    reason_codes: &mut Vec<PlanReasonCode>,
    messages: &mut Vec<String>,
) {
    for constraint in &family.split_constraints {
        if constraint.forbidden_boundaries.contains(&layer_boundary)
            || (constraint.reject_boundary_inside
                && constraint.range.contains_boundary(layer_boundary))
        {
            *decision = BoundaryDecision::Rejected;
            reason_codes.push(match constraint.kind {
                SplitConstraintKind::SharedKvProducerConsumer => PlanReasonCode::SharedKvRegionCut,
            });
            messages.push(constraint.reason.clone());
        }
    }

    for sideband in &family.sidebands {
        if layer_boundary <= sideband.first_required_layer {
            reason_codes.push(match sideband.kind {
                SidebandKind::TokenIds => PlanReasonCode::TokenSidebandRequired,
                SidebandKind::Rwkv7VFirst | SidebandKind::Gemma3nAltup => {
                    PlanReasonCode::ActivationSidebandRequired
                }
            });
            messages.push(sideband.reason.clone());
        }
    }
}

fn activation_payload_multiplier_for_boundary(
    family: &FamilyCapabilityRecord,
    layer_boundary: u32,
) -> u64 {
    let has_gemma3n_altup_sideband = family.sidebands.iter().any(|sideband| {
        sideband.kind == SidebandKind::Gemma3nAltup
            && layer_boundary <= sideband.first_required_layer
    });
    if has_gemma3n_altup_sideband {
        return 4;
    }

    let has_rwkv7_v_first_sideband = family.sidebands.iter().any(|sideband| {
        sideband.kind == SidebandKind::Rwkv7VFirst
            && layer_boundary <= sideband.first_required_layer
    });
    if has_rwkv7_v_first_sideband { 2 } else { 1 }
}

pub fn wire_payload_bytes_per_token(activation_width: u32) -> u64 {
    u64::from(activation_width) * 4
}

fn diagnostics_for(
    stages: &[StagePlan],
    boundaries: &[BoundaryPlan],
    placement_signals: &[NodePlacementSignal],
    family: Option<&FamilyCapabilityRecord>,
    policy: PlannerPolicy,
) -> Vec<PlanDiagnostic> {
    let mut diagnostics = Vec::new();
    artifact_diagnostics::append_artifact_diagnostics(&mut diagnostics, stages, placement_signals);
    for stage in stages {
        if matches!(
            stage.migration_policy,
            MigrationPolicy::StickyRecurrentOwner
        ) {
            diagnostics.push(PlanDiagnostic {
                severity: DiagnosticSeverity::Info,
                code: PlanReasonCode::RecurrentOwnerSticky,
                message: format!(
                    "{} owns recurrent state for layers {}..{}; route future tokens back to {} and only transfer activations across stage boundaries",
                    stage.stage_id, stage.layer_start, stage.layer_end, stage.node_id
                ),
            });
        }
    }

    if policy.allow_recurrent_state_transfer {
        diagnostics.push(PlanDiagnostic {
            severity: DiagnosticSeverity::Warning,
            code: PlanReasonCode::RecurrentStateTransferAllowed,
            message: "recurrent state transfer is enabled; this should be reserved for explicit recompute-or-transfer flows, not normal routing".to_string(),
        });
    }

    if let Some(family) = family {
        match family.exact_state_mobility {
            ExactStateMobility::Accepted => diagnostics.push(PlanDiagnostic {
                severity: DiagnosticSeverity::Info,
                code: PlanReasonCode::ExactStateMobilityAccepted,
                message: format!(
                    "{} exact state mobility is within current payload policy",
                    family.family_id
                ),
            }),
            ExactStateMobility::RejectedTooLarge => diagnostics.push(PlanDiagnostic {
                severity: DiagnosticSeverity::Warning,
                code: PlanReasonCode::ExactStateMobilityRejected,
                message: format!(
                    "{} exact state mobility is rejected for normal routing; route activations and keep live state sticky",
                    family.family_id
                ),
            }),
            ExactStateMobility::Untested => {}
        }
    }

    for boundary in boundaries {
        if boundary.decision == BoundaryDecision::Rejected {
            diagnostics.push(PlanDiagnostic {
                severity: DiagnosticSeverity::Error,
                code: boundary
                    .reason_codes
                    .iter()
                    .copied()
                    .find(|code| *code == PlanReasonCode::SharedKvRegionCut)
                    .unwrap_or(PlanReasonCode::RecurrentStateTransferRejected),
                message: format!(
                    "boundary at layer {} is rejected: {}",
                    boundary.layer_boundary,
                    boundary.messages.join("; ")
                ),
            });
        }
    }

    diagnostics
}

#[cfg(test)]
mod balanced_split_tests {
    use super::*;

    /// The exhaustive search the dynamic program replaced: every ascending
    /// combination, lowest score first, then the lexicographically smallest.
    fn exhaustive_balanced_splits(
        accepted: &[u32],
        layer_start: u32,
        layer_end: u32,
        stage_count: usize,
    ) -> Option<Vec<u32>> {
        // The score is restated here instead of reusing `BalancedSplitCost`,
        // so a mistake in that formula cannot hide on both sides.
        let span = u128::from(layer_end - layer_start);
        let score_at = |position: usize, boundary: u32| {
            let actual = u128::from(boundary - layer_start) * stage_count as u128;
            let ideal = span * (position + 1) as u128;
            actual.abs_diff(ideal)
        };
        let split_count = stage_count - 1;
        (0u32..1 << accepted.len())
            .filter(|mask| mask.count_ones() as usize == split_count)
            .map(|mask| {
                let splits = (0..accepted.len())
                    .filter(|index| mask & (1 << index) != 0)
                    .map(|index| accepted[index])
                    .collect::<Vec<_>>();
                let score = splits
                    .iter()
                    .enumerate()
                    .map(|(position, boundary)| score_at(position, *boundary))
                    .sum::<u128>();
                (score, splits)
            })
            .min()
            .map(|(_, splits)| splits)
    }

    #[test]
    fn matches_exhaustive_search_on_every_small_accepted_set() {
        let layer_start = 3;
        let layer_end = 14;
        let candidates = (layer_start + 1..layer_end).collect::<Vec<_>>();
        for mask in 0u32..1 << candidates.len() {
            let accepted = candidates
                .iter()
                .enumerate()
                .filter(|(index, _)| mask & (1 << index) != 0)
                .map(|(_, boundary)| *boundary)
                .collect::<Vec<_>>();
            for stage_count in 2..=6 {
                assert_eq!(
                    choose_balanced_splits(&accepted, layer_start, layer_end, stage_count),
                    exhaustive_balanced_splits(&accepted, layer_start, layer_end, stage_count),
                    "accepted {accepted:?}, {stage_count} stages"
                );
            }
        }
    }

    #[test]
    fn matches_exhaustive_search_on_sparse_boundaries_over_wide_ranges() {
        // Deterministic xorshift so failures reproduce.
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };
        for _ in 0..400 {
            let layer_start = next(1_000) as u32;
            let layer_end = layer_start + 20 + next(4_096) as u32;
            let wanted = next(15) as usize;
            let mut accepted = std::collections::BTreeSet::new();
            while accepted.len() < wanted {
                accepted
                    .insert(layer_start + 1 + next(u64::from(layer_end - layer_start - 1)) as u32);
            }
            let accepted = accepted.into_iter().collect::<Vec<_>>();
            let stage_count = 2 + next(8) as usize;
            assert_eq!(
                choose_balanced_splits(&accepted, layer_start, layer_end, stage_count),
                exhaustive_balanced_splits(&accepted, layer_start, layer_end, stage_count),
                "accepted {accepted:?} over {layer_start}..{layer_end}, {stage_count} stages"
            );
        }
    }

    #[test]
    fn breaks_score_ties_toward_earlier_boundaries() {
        // Ideal boundary for 2 stages over 0..10 is 5; 4 and 6 tie.
        assert_eq!(choose_balanced_splits(&[4, 6], 0, 10, 2), Some(vec![4]));
    }

    #[test]
    fn rejects_too_few_accepted_boundaries() {
        assert_eq!(choose_balanced_splits(&[4, 6], 0, 10, 4), None);
        assert_eq!(choose_balanced_splits(&[], 0, 10, 2), None);
        assert_eq!(choose_balanced_splits(&[4], 0, 10, 1), None);
    }
}

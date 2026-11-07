use crate::{BoundaryPlan, FamilyCapabilityRecord, LayerRange, TopologyPlanRequest};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    EmptyLayers,
    EmptyNodes,
    TooManyLayers {
        found: usize,
    },
    NonContiguousLayers {
        expected: u32,
        found: u32,
    },
    InvalidSplitBoundary {
        boundary: u32,
        layer_start: u32,
        layer_end: u32,
    },
    NonAscendingSplitBoundary {
        previous: u32,
        boundary: u32,
    },
    NotEnoughNodesForSplits {
        stages: usize,
        nodes: usize,
    },
    InvalidStageCount {
        stages: usize,
    },
    NoAcceptedBalancedSplit {
        stages: usize,
    },
    FamilyLayerCountMismatch {
        family_id: String,
        expected: u32,
        found: u32,
    },
    InvalidFamilyRange {
        family_id: String,
        field: &'static str,
        start: u32,
        end: u32,
        layer_count: u32,
    },
    InvalidFamilySideband {
        family_id: String,
        first_required_layer: u32,
        layer_count: u32,
    },
    StageParameterBytesOverflow {
        stage_index: usize,
    },
    /// The planned cut crosses a boundary the family capability rejects.
    /// Carries every rejected boundary with its reason codes and messages.
    RejectedBoundaries {
        boundaries: Vec<BoundaryPlan>,
    },
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyLayers => write!(f, "topology plan requires at least one layer"),
            Self::EmptyNodes => write!(f, "topology plan requires at least one node"),
            Self::TooManyLayers { found } => write!(
                f,
                "topology plan has {found} layers; at most {} are supported",
                u32::MAX
            ),
            Self::NonContiguousLayers { expected, found } => write!(
                f,
                "layers must be sorted and contiguous from 0: expected layer {expected}, found {found}"
            ),
            Self::InvalidSplitBoundary {
                boundary,
                layer_start,
                layer_end,
            } => write!(
                f,
                "invalid split boundary {boundary}; expected {layer_start} < boundary < {layer_end}"
            ),
            Self::NonAscendingSplitBoundary { previous, boundary } => write!(
                f,
                "split boundaries must be strictly ascending: previous {previous}, found {boundary}"
            ),
            Self::NotEnoughNodesForSplits { stages, nodes } => write!(
                f,
                "split plan requires {stages} nodes but only {nodes} were provided"
            ),
            Self::InvalidStageCount { stages } => {
                write!(
                    f,
                    "balanced split selection requires at least two stages, got {stages}"
                )
            }
            Self::NoAcceptedBalancedSplit { stages } => write!(
                f,
                "no product-approved balanced split exists for {stages} stages"
            ),
            Self::FamilyLayerCountMismatch {
                family_id,
                expected,
                found,
            } => write!(
                f,
                "family capability {family_id} expects {expected} layers, found {found}"
            ),
            Self::InvalidFamilyRange {
                family_id,
                field,
                start,
                end,
                layer_count,
            } => write!(
                f,
                "family capability {family_id} has invalid {field} range {start}..{end}; expected a non-empty range within 0..{layer_count}"
            ),
            Self::InvalidFamilySideband {
                family_id,
                first_required_layer,
                layer_count,
            } => write!(
                f,
                "family capability {family_id} requires a sideband through layer {first_required_layer}, past its {layer_count} layers"
            ),
            Self::StageParameterBytesOverflow { stage_index } => write!(
                f,
                "stage {stage_index} parameter bytes overflow a 64-bit total"
            ),
            Self::RejectedBoundaries { boundaries } => {
                write!(f, "topology plan crosses rejected boundaries: ")?;
                for (index, boundary) in boundaries.iter().enumerate() {
                    if index > 0 {
                        write!(f, "; ")?;
                    }
                    write!(
                        f,
                        "layer {}: {}",
                        boundary.layer_boundary,
                        boundary.messages.join("; ")
                    )?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for PlanError {}

/// Validates a whole-model planning request and returns its layer count.
///
/// Layers must be indexed `0..layer_count` in order, so every plan covers the
/// model from its first layer to its last, and a family capability must
/// describe exactly those layers.
pub(crate) fn validate_request(request: &TopologyPlanRequest) -> Result<u32, PlanError> {
    if request.layers.is_empty() {
        return Err(PlanError::EmptyLayers);
    }
    if request.nodes.is_empty() {
        return Err(PlanError::EmptyNodes);
    }
    let layer_count =
        u32::try_from(request.layers.len()).map_err(|_| PlanError::TooManyLayers {
            found: request.layers.len(),
        })?;

    for (expected, layer) in (0..layer_count).zip(request.layers.iter()) {
        if layer.index != expected {
            return Err(PlanError::NonContiguousLayers {
                expected,
                found: layer.index,
            });
        }
    }

    if let Some(family) = &request.family {
        validate_family(family, layer_count)?;
    }

    Ok(layer_count)
}

fn validate_family(family: &FamilyCapabilityRecord, layer_count: u32) -> Result<(), PlanError> {
    if family.layer_count != layer_count {
        return Err(PlanError::FamilyLayerCountMismatch {
            family_id: family.family_id.clone(),
            expected: family.layer_count,
            found: layer_count,
        });
    }
    for range in &family.recurrent_ranges {
        validate_family_range(family, "recurrent", *range)?;
    }
    for constraint in &family.split_constraints {
        validate_family_range(family, "split constraint", constraint.range)?;
    }
    for sideband in &family.sidebands {
        if sideband.first_required_layer > family.layer_count {
            return Err(PlanError::InvalidFamilySideband {
                family_id: family.family_id.clone(),
                first_required_layer: sideband.first_required_layer,
                layer_count: family.layer_count,
            });
        }
    }
    Ok(())
}

fn validate_family_range(
    family: &FamilyCapabilityRecord,
    field: &'static str,
    range: LayerRange,
) -> Result<(), PlanError> {
    if range.start < range.end && range.end <= family.layer_count {
        return Ok(());
    }
    Err(PlanError::InvalidFamilyRange {
        family_id: family.family_id.clone(),
        field,
        start: range.start,
        end: range.end,
        layer_count: family.layer_count,
    })
}

/// Fails a finished plan whose cut crosses a boundary the family rejects.
pub(crate) fn reject_rejected_boundaries(
    plan: crate::TopologyPlan,
) -> Result<crate::TopologyPlan, PlanError> {
    let rejected = plan
        .boundaries
        .iter()
        .filter(|boundary| boundary.decision == crate::BoundaryDecision::Rejected)
        .cloned()
        .collect::<Vec<_>>();
    if rejected.is_empty() {
        Ok(plan)
    } else {
        Err(PlanError::RejectedBoundaries {
            boundaries: rejected,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        LayerSpec, NodeSpec, PlannerPolicy, SidebandKind, SidebandRequirement, SplitConstraint,
        SplitConstraintKind, TEST_LLAMA_ARCHITECTURE_CATALOG, dense_attention_layers,
        gemma3n_capability, gemma4_e4b_capability, infer_family_capability, kimi_linear_capability,
        plan_balanced_accepted_contiguous, plan_contiguous_with_splits, plan_even_contiguous,
        plan_package_aware_contiguous, plan_weighted_contiguous, qwen3next_capability,
        rwkv7_capability,
    };

    fn request(layers: Vec<LayerSpec>, node_count: usize) -> TopologyPlanRequest {
        TopologyPlanRequest {
            topology_id: "validation".to_string(),
            model_id: "validation".to_string(),
            layers,
            nodes: (0..node_count)
                .map(|index| NodeSpec {
                    node_id: format!("node-{index}"),
                    cached_slice_bytes: 0,
                    vram_bytes: 0,
                })
                .collect(),
            family: None,
            policy: PlannerPolicy::default(),
        }
    }

    fn layer(index: u32) -> LayerSpec {
        LayerSpec {
            index,
            attention: true,
            recurrent: false,
            parameter_bytes: 1,
        }
    }

    #[test]
    fn layers_must_start_at_zero() {
        let offset = request((4..8).map(layer).collect(), 2);
        assert_eq!(
            plan_even_contiguous(&offset),
            Err(PlanError::NonContiguousLayers {
                expected: 0,
                found: 4
            })
        );
    }

    #[test]
    fn last_layer_index_cannot_overflow_the_layer_end() {
        let request = request(vec![layer(u32::MAX)], 1);
        assert_eq!(
            plan_even_contiguous(&request),
            Err(PlanError::NonContiguousLayers {
                expected: 0,
                found: u32::MAX
            })
        );
        assert_eq!(
            plan_contiguous_with_splits(&request, &[]),
            Err(PlanError::NonContiguousLayers {
                expected: 0,
                found: u32::MAX
            })
        );
    }

    #[test]
    fn stage_parameter_bytes_overflow_is_an_error() {
        let mut layers = dense_attention_layers(4, u64::MAX);
        layers[3].parameter_bytes = 1;
        let request = request(layers, 2);
        assert_eq!(
            plan_contiguous_with_splits(&request, &[3]),
            Err(PlanError::StageParameterBytesOverflow { stage_index: 0 })
        );
    }

    #[test]
    fn balanced_selection_reports_overflow_instead_of_no_split() {
        let request = request(dense_attention_layers(4, u64::MAX), 2);
        assert!(matches!(
            plan_balanced_accepted_contiguous(&request, 2),
            Err(PlanError::StageParameterBytesOverflow { .. })
        ));
    }

    #[test]
    fn every_planner_fails_a_plan_across_a_rejected_boundary() {
        let mut request = request(dense_attention_layers(42, 10), 3);
        request.family = Some(gemma4_e4b_capability(42, 2560));
        let rejected_layers = |result: Result<crate::TopologyPlan, PlanError>| match result {
            Err(PlanError::RejectedBoundaries { boundaries }) => boundaries
                .iter()
                .map(|boundary| boundary.layer_boundary)
                .collect::<Vec<_>>(),
            other => panic!("expected rejected boundaries, got {other:?}"),
        };

        // Without node weights or package signals every planner cuts 14/28.
        assert_eq!(rejected_layers(plan_even_contiguous(&request)), [14, 28]);
        assert_eq!(
            rejected_layers(plan_weighted_contiguous(&request)),
            [14, 28]
        );
        assert_eq!(
            rejected_layers(plan_package_aware_contiguous(&request)),
            [14, 28]
        );

        // A cached slice takes the package-aware ordering path.
        request.nodes[0].cached_slice_bytes = 1;
        assert_eq!(
            rejected_layers(plan_package_aware_contiguous(&request)),
            [14, 28]
        );
    }

    #[test]
    fn node_vram_total_does_not_overflow() {
        let mut request = request(dense_attention_layers(4, 1), 2);
        for node in &mut request.nodes {
            node.vram_bytes = u64::MAX;
        }
        let plan = plan_weighted_contiguous(&request).expect("plan");
        assert_eq!(
            plan.stages
                .iter()
                .map(|stage| (stage.layer_start, stage.layer_end))
                .collect::<Vec<_>>(),
            vec![(0, 2), (2, 4)]
        );
    }

    #[test]
    fn family_recurrent_range_must_lie_within_the_model() {
        let mut request = request(dense_attention_layers(4, 1), 2);
        request.family = Some(qwen3next_capability(
            4,
            64,
            vec![LayerRange { start: 2, end: 5 }],
        ));
        assert_eq!(
            plan_even_contiguous(&request),
            Err(PlanError::InvalidFamilyRange {
                family_id: "qwen3next".to_string(),
                field: "recurrent",
                start: 2,
                end: 5,
                layer_count: 4,
            })
        );
    }

    #[test]
    fn family_split_constraint_range_must_be_non_empty() {
        let mut family = gemma3n_capability(4, 64);
        family.split_constraints = vec![SplitConstraint {
            kind: SplitConstraintKind::SharedKvProducerConsumer,
            range: LayerRange { start: 3, end: 1 },
            forbidden_boundaries: Vec::new(),
            reject_boundary_inside: true,
            reason: "reversed".to_string(),
        }];
        let mut request = request(dense_attention_layers(4, 1), 2);
        request.family = Some(family);
        assert!(matches!(
            plan_even_contiguous(&request),
            Err(PlanError::InvalidFamilyRange {
                field: "split constraint",
                start: 3,
                end: 1,
                ..
            })
        ));
    }

    #[test]
    fn family_sideband_must_end_within_the_model() {
        let mut family = rwkv7_capability(4, 64);
        family.sidebands = vec![SidebandRequirement {
            kind: SidebandKind::Rwkv7VFirst,
            first_required_layer: 5,
            reason: "past the model".to_string(),
        }];
        let mut request = request(dense_attention_layers(4, 1), 2);
        request.family = Some(family);
        assert_eq!(
            plan_even_contiguous(&request),
            Err(PlanError::InvalidFamilySideband {
                family_id: "rwkv7".to_string(),
                first_required_layer: 5,
                layer_count: 4,
            })
        );
    }

    #[test]
    fn built_in_family_capabilities_pass_validation() {
        for layer_count in [1, 2, 3, 4, 8, 20, 21, 42, 64] {
            let mut families = TEST_LLAMA_ARCHITECTURE_CATALOG
                .iter()
                .filter_map(|entry| {
                    infer_family_capability(entry.llama_architecture, layer_count, 4096)
                })
                .collect::<Vec<_>>();
            families.push(gemma3n_capability(layer_count, 4096));
            families.push(gemma4_e4b_capability(layer_count, 4096));
            families.push(kimi_linear_capability(layer_count, 4096));
            for family in families {
                assert_eq!(
                    validate_family(&family, layer_count),
                    Ok(()),
                    "{} at {layer_count} layers",
                    family.family_id
                );
            }
        }
    }
}

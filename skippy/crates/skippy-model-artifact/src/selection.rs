//! Artifact preference shared by the Mesh and standalone CLIs.

use std::cmp::Ordering;

use skippy_model_ref::{complete_split_gguf_shard_set, split_gguf_shard_info};

use crate::ModelArtifactFile;

pub fn repo_prefers_gguf_only(repo: &str) -> bool {
    repo.to_ascii_lowercase().contains("gguf")
}

pub fn file_preference_score(file: &str) -> usize {
    if file.contains("-00001-of-") {
        return 0;
    }
    const PREFERRED: &[&str] = &[
        "Q4_K_M", "Q4_K_S", "Q4_1", "Q5_K_M", "Q5_K_S", "Q8_0", "BF16",
    ];
    PREFERRED
        .iter()
        .position(|needle| file.contains(needle))
        .map(|pos| pos + 1)
        .unwrap_or(PREFERRED.len() + 2)
}

/// Total bytes of the GGUF variant `file`, summed over every shard of a split
/// GGUF. `None` when a size is unknown, the shard set is incomplete, or the
/// sum overflows.
pub fn gguf_variant_size_bytes(file: &str, siblings: &[ModelArtifactFile]) -> Option<u64> {
    if split_gguf_shard_info(file).is_some() {
        return complete_split_gguf_shard_set(file, siblings, |candidate| &candidate.path)?
            .into_iter()
            .try_fold(0u64, |total, shard| total.checked_add(shard.size_bytes?));
    }
    siblings
        .iter()
        .find(|candidate| candidate.path == file)
        .and_then(|candidate| candidate.size_bytes)
}

pub fn compare_gguf_candidates_by_fit(
    left_file: &str,
    left_size: Option<u64>,
    right_file: &str,
    right_size: Option<u64>,
    available_bytes: u64,
) -> Ordering {
    if available_bytes > 0 {
        match (left_size, right_size) {
            (Some(left), Some(right)) => {
                let left_bucket = fit_bucket(left, available_bytes);
                let right_bucket = fit_bucket(right, available_bytes);
                if left_bucket != right_bucket {
                    return left_bucket.cmp(&right_bucket);
                }
                let size_order = if left_bucket <= 1 {
                    right.cmp(&left)
                } else {
                    left.cmp(&right)
                };
                if size_order != Ordering::Equal {
                    return size_order;
                }
            }
            (Some(_), None) => return Ordering::Less,
            (None, Some(_)) => return Ordering::Greater,
            (None, None) => {}
        }
    }
    file_preference_score(left_file)
        .cmp(&file_preference_score(right_file))
        .then_with(|| left_file.cmp(right_file))
}

fn fit_bucket(size_bytes: u64, available_bytes: u64) -> u8 {
    if size_bytes.saturating_mul(10) <= available_bytes.saturating_mul(9) {
        0
    } else if size_bytes.saturating_mul(10) <= available_bytes.saturating_mul(11) {
        1
    } else {
        2
    }
}

pub fn select_default_gguf_file(
    files: &[ModelArtifactFile],
    available_bytes: u64,
) -> Option<ModelArtifactFile> {
    let mut candidates = files
        .iter()
        .filter(|file| {
            let basename = file.path.rsplit('/').next().unwrap_or(&file.path);
            let lower = basename.to_ascii_lowercase();
            lower.ends_with(".gguf")
                && !lower.starts_with("mmproj")
                && split_gguf_shard_info(&file.path).is_none_or(|shard| {
                    shard.part == "00001"
                        && complete_split_gguf_shard_set(&file.path, files, |file| &file.path)
                            .is_some()
                })
        })
        .map(|file| (file.clone(), gguf_variant_size_bytes(&file.path, files)))
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        compare_gguf_candidates_by_fit(
            &left.0.path,
            left.1,
            &right.0.path,
            right.1,
            available_bytes,
        )
    });
    candidates.into_iter().next().map(|(file, _)| file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_largest_fitting_variant_using_mesh_buckets() {
        let files = [
            ModelArtifactFile {
                path: "model-Q4_K_M.gguf".into(),
                size_bytes: Some(4),
                sha256: None,
            },
            ModelArtifactFile {
                path: "model-Q8_0.gguf".into(),
                size_bytes: Some(8),
                sha256: None,
            },
            ModelArtifactFile {
                path: "model-BF16.gguf".into(),
                size_bytes: Some(16),
                sha256: None,
            },
            ModelArtifactFile {
                path: "mmproj.gguf".into(),
                size_bytes: Some(1),
                sha256: None,
            },
        ];
        assert_eq!(
            select_default_gguf_file(&files, 10).unwrap().path,
            "model-Q8_0.gguf"
        );
        assert_eq!(
            select_default_gguf_file(&files, 0).unwrap().path,
            "model-Q4_K_M.gguf"
        );
    }

    fn sized(path: &str, size_bytes: u64) -> ModelArtifactFile {
        ModelArtifactFile {
            path: path.into(),
            size_bytes: Some(size_bytes),
            sha256: None,
        }
    }

    #[test]
    fn split_variant_size_requires_every_shard() {
        let complete = [
            sized("Q4/model-00001-of-00002.gguf", 3),
            sized("Q4/model-00002-of-00002.gguf", 4),
        ];
        assert_eq!(
            gguf_variant_size_bytes("Q4/model-00001-of-00002.gguf", &complete),
            Some(7)
        );

        let incomplete = [
            sized("Q4/model-00001-of-00003.gguf", 3),
            sized("Q4/model-00003-of-00003.gguf", 4),
        ];
        assert_eq!(
            gguf_variant_size_bytes("Q4/model-00001-of-00003.gguf", &incomplete),
            None
        );
    }

    #[test]
    fn split_variant_size_rejects_overflowing_shard_sizes() {
        let files = [
            sized("model-00001-of-00002.gguf", u64::MAX),
            sized("model-00002-of-00002.gguf", 1),
        ];
        assert_eq!(
            gguf_variant_size_bytes("model-00001-of-00002.gguf", &files),
            None
        );
    }

    #[test]
    fn default_selection_skips_incomplete_split_sets() {
        let files = [
            sized("BF16/model-BF16-00001-of-00003.gguf", 1),
            sized("BF16/model-BF16-00002-of-00003.gguf", 1),
            sized("model-Q8_0.gguf", 8),
        ];
        assert_eq!(
            select_default_gguf_file(&files, 0).unwrap().path,
            "model-Q8_0.gguf"
        );
    }
}

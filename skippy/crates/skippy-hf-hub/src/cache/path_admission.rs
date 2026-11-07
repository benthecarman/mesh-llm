//! Admit names before they become path components under the cache.
//!
//! File names and revisions reach the cache layout from callers and from the
//! Hub itself (tree listings, `X-Repo-Commit`). Every one of them is joined
//! onto a cache or local directory, so each must be a relative path made only
//! of plain names: no root, drive prefix, `.` or `..` may move the joined path
//! outside the directory it was joined onto.

use std::path::{Component, Path};

use crate::error::{HFError, HFResult};

/// Whether `value` stays below the directory it is joined onto: it is
/// non-empty and every component is [`Component::Normal`].
fn is_confined_relative(value: &str) -> bool {
    let mut components = Path::new(value).components().peekable();
    components.peek().is_some() && components.all(|c| matches!(c, Component::Normal(_)))
}

/// Admit a repo-relative file name before it is joined into a snapshot,
/// `.no_exist`, or local-dir path.
pub(crate) fn admit_repo_file_name(filename: &str) -> HFResult<()> {
    if !is_confined_relative(filename) {
        return Err(HFError::InvalidParameter(format!(
            "repository file name {filename:?} is not a confined relative path"
        )));
    }
    Ok(())
}

/// Admit the repo folder and revision of a `refs/` entry. Readers and writers
/// share this check, so a ref that cannot be read back is never written.
///
/// Each `/`-separated revision segment must be one plain name without the
/// characters Git also forbids in ref names (`\`, `:`, control characters),
/// so the same revision maps to the same file on every platform.
pub(crate) fn admit_ref_location(repo_folder: &str, revision: &str) -> HFResult<()> {
    let plain_segment = |value: &str| {
        is_confined_relative(value)
            && !value
                .chars()
                .any(|ch| ch.is_control() || matches!(ch, '/' | '\\' | ':'))
    };
    if !plain_segment(repo_folder) || !revision.split('/').all(plain_segment) {
        return Err(HFError::InvalidParameter(
            "unsafe cached reference path".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{admit_ref_location, admit_repo_file_name};

    #[test]
    fn repo_file_names_must_stay_below_the_snapshot() {
        for name in ["config.json", "subdir/model.bin", "a/b/c.gguf"] {
            assert!(admit_repo_file_name(name).is_ok(), "{name:?}");
        }
        for name in [
            "",
            ".",
            "..",
            "./config.json",
            "../escape",
            "subdir/../../escape",
            "subdir/..",
            "/etc/passwd",
        ] {
            assert!(admit_repo_file_name(name).is_err(), "{name:?}");
        }
    }

    #[cfg(windows)]
    #[test]
    fn repo_file_names_reject_windows_roots_and_prefixes() {
        for name in [
            r"..\escape",
            r"subdir\..\..\escape",
            r"C:\escape",
            "C:escape",
            r"\\server\share\escape",
            r"\escape",
        ] {
            assert!(admit_repo_file_name(name).is_err(), "{name:?}");
        }
    }

    #[test]
    fn ref_locations_must_be_plain_segments() {
        for revision in ["main", "v1.0", "refs/pr/1", "release/étape#v2"] {
            assert!(admit_ref_location("models--gpt2", revision).is_ok());
        }
        for revision in [
            "",
            ".",
            "..",
            "../escape",
            "refs/../../escape",
            "/absolute",
            "main/",
            "a//b",
            r"a\b",
            "a:b",
            "a\nb",
        ] {
            assert!(
                admit_ref_location("models--gpt2", revision).is_err(),
                "{revision:?}"
            );
        }
        for folder in ["", "..", "a/b", r"a\b"] {
            assert!(admit_ref_location(folder, "main").is_err(), "{folder:?}");
        }
    }
}

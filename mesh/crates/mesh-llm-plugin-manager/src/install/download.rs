//! Downloading a release asset and checking it is the archive the install
//! expects: the size GitHub reports and the digest the catalog pins (or,
//! without a pin, the digest GitHub reports).

use std::io::Write;

use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use reqwest::Client;
use sha2::Digest;
use tempfile::TempPath;

use super::{PluginProgressEvent, PluginProgressReporter};
use crate::{catalog_pin::CatalogPin, github::GitHubReleaseAsset};

/// The largest plugin release asset the installer downloads. A plugin archive
/// is an executable and its assets; this is far above any real one and keeps
/// a hostile or broken release from filling the disk.
const MAX_PLUGIN_ASSET_BYTES: u64 = 512 * 1024 * 1024;

/// Download the asset and verify it against the catalog pin when there is one,
/// and against the digest GitHub reports when there is not. Returns the archive
/// path and the digest the bytes were verified against. The archive is
/// deleted when the returned path is dropped, so a failed install leaves
/// nothing behind.
pub(super) async fn download_asset(
    client: &Client,
    asset: &GitHubReleaseAsset,
    pin: Option<&CatalogPin>,
    progress: &mut impl PluginProgressReporter,
) -> Result<(TempPath, String)> {
    let expected_sha256 = expected_asset_sha256(asset, pin)?;
    let expected_size = expected_asset_size(asset)?;
    progress.report(PluginProgressEvent::DownloadStarted {
        asset: asset.name.clone(),
        total_bytes: Some(expected_size),
    });
    let response = client
        .get(&asset.browser_download_url)
        .header(reqwest::header::USER_AGENT, crate::github::USER_AGENT)
        .send()
        .await
        .with_context(|| format!("download plugin asset {}", asset.name))?;
    let status = response.status();
    if !status.is_success() {
        bail!("plugin asset download failed: {status} {}", asset.name);
    }

    let mut temp = tempfile::Builder::new()
        .prefix("mesh-plugin-asset-")
        .suffix(&format!("-{}", asset.name))
        .tempfile()
        .context("create plugin asset temp file")?;

    let mut downloaded = 0u64;
    let mut hasher = sha2::Sha256::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("read plugin asset {}", asset.name))?;
        downloaded = downloaded.saturating_add(chunk.len() as u64);
        if downloaded > expected_size {
            bail!(
                "plugin asset {} is larger than the {expected_size} bytes GitHub reports",
                asset.name
            );
        }
        temp.write_all(&chunk)
            .with_context(|| format!("write plugin asset temp file {}", temp.path().display()))?;
        hasher.update(&chunk);
        progress.report(PluginProgressEvent::DownloadProgress {
            downloaded_bytes: downloaded,
            total_bytes: Some(expected_size),
        });
    }
    if downloaded != expected_size {
        bail!(
            "plugin asset {} ended after {downloaded} of the {expected_size} bytes GitHub reports",
            asset.name
        );
    }
    temp.flush()
        .with_context(|| format!("flush plugin asset temp file {}", temp.path().display()))?;
    verify_asset_sha256(&asset.name, &expected_sha256, hasher)?;
    progress.report(PluginProgressEvent::DownloadFinished {
        asset: asset.name.clone(),
    });
    Ok((temp.into_temp_path(), expected_sha256))
}

/// The number of bytes the download must be. GitHub reports every asset's
/// size, so a release without one, or with one over the installer's limit, is
/// refused before anything is downloaded.
fn expected_asset_size(asset: &GitHubReleaseAsset) -> Result<u64> {
    let size = asset
        .size
        .with_context(|| format!("plugin asset {} is missing its GitHub size", asset.name))?;
    if size > MAX_PLUGIN_ASSET_BYTES {
        bail!(
            "plugin asset {} is {size} bytes, over the {MAX_PLUGIN_ASSET_BYTES}-byte limit",
            asset.name
        );
    }
    Ok(size)
}

/// The digest GitHub reports for an asset, normalized to lowercase hex. GitHub
/// reports it on the same mutable release object as the asset itself, so on a
/// pinned install it is a cross-check and never the authority.
pub(super) fn reported_asset_sha256(asset: &GitHubReleaseAsset) -> Result<Option<String>> {
    let Some(digest) = asset.digest.as_deref() else {
        return Ok(None);
    };
    let Some(sha256) = digest.trim().strip_prefix("sha256:") else {
        bail!("plugin asset {} digest must use sha256", asset.name);
    };
    let sha256 = sha256.to_ascii_lowercase();
    if sha256.len() != 64 || !sha256.chars().all(|ch| ch.is_ascii_hexdigit()) {
        bail!("plugin asset {} has an invalid sha256 digest", asset.name);
    }
    Ok(Some(sha256))
}

fn required_plugin_asset_sha256(asset: &GitHubReleaseAsset) -> Result<String> {
    reported_asset_sha256(asset)?
        .with_context(|| format!("plugin asset {} is missing a GitHub digest", asset.name))
}

/// The digest a downloaded archive must match. A catalog install matches the
/// catalog's pin: the digest GitHub reports describes whatever the release
/// holds now, and whoever controls the repository can rewrite both. A GitHub
/// reference names no catalog entry, so GitHub's own digest is all there is.
fn expected_asset_sha256(asset: &GitHubReleaseAsset, pin: Option<&CatalogPin>) -> Result<String> {
    match pin {
        Some(pin) => Ok(pin.sha256().to_string()),
        None => required_plugin_asset_sha256(asset),
    }
}

fn verify_asset_sha256(asset_name: &str, expected: &str, hasher: sha2::Sha256) -> Result<()> {
    let actual = hex::encode(hasher.finalize());
    if actual != expected {
        bail!("plugin asset checksum mismatch for {asset_name}: expected {expected}, got {actual}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        io::{BufRead, BufReader, Write},
        net::TcpListener,
        thread,
    };

    use super::*;

    /// A catalog entry that pins `v1.0.0` and one platform digest.
    const PINNED_CATALOG_LINE: &str = r#"{"name":"demo","description":"Demo","github_url":"https://github.com/mesh-llm/demo","author_email":"dev@example.com","author_name":"Mesh LLM","version":"v1.0.0","sha256":{"aarch64-apple-darwin":"abababababababababababababababababababababababababababababababab"}}"#;

    fn pinned_catalog_entry() -> crate::catalog::CatalogEntry {
        crate::catalog::PluginCatalog::parse_jsonl(PINNED_CATALOG_LINE)
            .unwrap()
            .find_exact("demo")
            .unwrap()
            .clone()
    }

    #[test]
    fn modified_plugin_asset_is_rejected_before_extraction() {
        let expected = hex::encode(sha2::Sha256::digest(b"expected plugin archive"));
        let asset = GitHubReleaseAsset {
            name: "demo.tar.gz".to_string(),
            browser_download_url: "https://example.invalid/demo.tar.gz".to_string(),
            size: None,
            digest: Some(format!("sha256:{expected}")),
        };
        let mut hasher = sha2::Sha256::new();
        hasher.update(b"modified plugin archive");

        let error = verify_asset_sha256(&asset.name, &expected, hasher)
            .expect_err("modified plugin asset must fail verification");
        assert!(error.to_string().contains("checksum mismatch"), "{error:?}");
    }

    /// The attack the catalog pin exists for. The release was rewritten, so
    /// GitHub's own digest describes the replaced bytes and only the catalog
    /// still names the reviewed archive.
    #[test]
    fn a_pinned_install_verifies_the_catalog_digest_not_githubs() {
        let replaced = hex::encode(sha2::Sha256::digest(b"replaced plugin archive"));
        let asset = GitHubReleaseAsset {
            name: "demo.tar.gz".to_string(),
            browser_download_url: "https://example.invalid/demo.tar.gz".to_string(),
            size: None,
            digest: Some(format!("sha256:{replaced}")),
        };
        let pin = CatalogPin::for_entry(&pinned_catalog_entry(), "aarch64-apple-darwin").unwrap();

        assert_eq!(
            expected_asset_sha256(&asset, Some(&pin)).unwrap(),
            pin.sha256(),
            "a pinned install must not accept the digest a rewritten release reports"
        );
        assert_eq!(expected_asset_sha256(&asset, None).unwrap(), replaced);

        let mut hasher = sha2::Sha256::new();
        hasher.update(b"replaced plugin archive");
        let error = verify_asset_sha256(
            &asset.name,
            &expected_asset_sha256(&asset, Some(&pin)).unwrap(),
            hasher,
        )
        .expect_err("the replaced archive must fail the catalog pin");
        assert!(error.to_string().contains("checksum mismatch"), "{error:?}");
    }

    #[test]
    fn plugin_assets_without_valid_github_digests_are_rejected() {
        let mut asset = GitHubReleaseAsset {
            name: "demo.tar.gz".to_string(),
            browser_download_url: "https://example.invalid/demo.tar.gz".to_string(),
            size: None,
            digest: None,
        };
        let missing = required_plugin_asset_sha256(&asset).unwrap_err();
        assert!(missing.to_string().contains("missing a GitHub digest"));

        asset.digest = Some("sha512:abc".to_string());
        let wrong_algorithm = required_plugin_asset_sha256(&asset).unwrap_err();
        assert!(wrong_algorithm.to_string().contains("must use sha256"));

        asset.digest = Some("sha256:not-a-digest".to_string());
        let malformed = required_plugin_asset_sha256(&asset).unwrap_err();
        assert!(malformed.to_string().contains("invalid sha256"));
    }

    /// Serve `body` from a one-shot local server, so a download test never
    /// leaves the machine. Returns the URL the asset is served at.
    fn serve_asset(body: Vec<u8>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/demo.tar.gz", listener.local_addr().unwrap());
        thread::spawn(move || {
            let Ok((mut socket, _)) = listener.accept() else {
                return;
            };
            let Ok(clone) = socket.try_clone() else {
                return;
            };
            let mut request = BufReader::new(clone);
            let mut line = String::new();
            while request.read_line(&mut line).is_ok_and(|read| read > 2) {
                line.clear();
            }
            let header = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/gzip\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                body.len()
            );
            let _ = socket.write_all(header.as_bytes());
            let _ = socket.write_all(&body);
            let _ = socket.flush();
        });
        url
    }

    /// An asset GitHub describes as `size` bytes with `body`'s digest, served
    /// from a local server that sends `body`.
    fn served_asset(body: &[u8], size: Option<u64>) -> GitHubReleaseAsset {
        GitHubReleaseAsset {
            name: "demo.tar.gz".to_string(),
            browser_download_url: serve_asset(body.to_vec()),
            size,
            digest: Some(format!(
                "sha256:{}",
                hex::encode(sha2::Sha256::digest(body))
            )),
        }
    }

    async fn download(asset: &GitHubReleaseAsset) -> Result<(TempPath, String)> {
        download_asset(&Client::new(), asset, None, &mut |_event| {}).await
    }

    #[tokio::test]
    async fn an_asset_of_the_reported_size_and_digest_downloads() {
        let body = b"plugin archive bytes".to_vec();
        let asset = served_asset(&body, Some(body.len() as u64));
        let (path, _digest) = download(&asset).await.unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), body);
        let kept = path.to_path_buf();
        drop(path);
        assert!(!kept.exists(), "dropping the archive path deletes it");
    }

    /// A release whose asset streams more than GitHub reports is cut off at
    /// the reported size, before the digest check, so it cannot fill the disk
    /// on the way to failing that check.
    #[tokio::test]
    async fn an_asset_longer_than_its_reported_size_is_refused() {
        let asset = served_asset(&[7u8; 4096], Some(16));
        let error = download(&asset).await.unwrap_err();
        assert!(
            error.to_string().contains("larger than the 16 bytes"),
            "{error:#}"
        );
    }

    #[tokio::test]
    async fn an_asset_shorter_than_its_reported_size_is_refused() {
        let asset = served_asset(b"short", Some(4096));
        let error = download(&asset).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("ended after 5 of the 4096 bytes"),
            "{error:#}"
        );
    }

    #[test]
    fn an_asset_without_a_size_or_over_the_limit_is_refused_before_download() {
        let mut asset = GitHubReleaseAsset {
            name: "demo.tar.gz".to_string(),
            browser_download_url: "https://example.invalid/demo.tar.gz".to_string(),
            size: None,
            digest: None,
        };
        let missing = expected_asset_size(&asset).unwrap_err();
        assert!(missing.to_string().contains("missing its GitHub size"));

        asset.size = Some(MAX_PLUGIN_ASSET_BYTES + 1);
        let oversized = expected_asset_size(&asset).unwrap_err();
        assert!(oversized.to_string().contains("-byte limit"), "{oversized}");

        asset.size = Some(MAX_PLUGIN_ASSET_BYTES);
        assert_eq!(expected_asset_size(&asset).unwrap(), MAX_PLUGIN_ASSET_BYTES);
    }
}

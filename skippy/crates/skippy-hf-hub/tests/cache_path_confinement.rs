//! Remote and caller-supplied file names must not escape the cache or local directory.
//!
//! A Hub tree listing is untrusted input: an entry such as `../escape.bin` would otherwise
//! be joined under `snapshots/<commit>/` (or `local_dir`) and written outside it. Downloads
//! must reject such names before they make any further request or touch the filesystem.

use std::path::Path;
use std::sync::{Arc, Mutex};

use hf_hub::{HFClient, HFError, HFResult};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const COMMIT: &str = "0123456789012345678901234567890123456789";
const HOSTILE_NAMES: [&str; 3] = ["../escape.bin", "a/../../escape.bin", "/escape.bin"];

/// Serve every request with a tree listing holding `name`, recording each request line.
async fn start_tree_server(name: &'static str, requests: Arc<Mutex<Vec<String>>>) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let requests = Arc::clone(&requests);
            tokio::spawn(async move {
                let mut request = vec![0; 4096];
                let Ok(read) = socket.read(&mut request).await else {
                    return;
                };
                let request = String::from_utf8_lossy(&request[..read]);
                let line = request.lines().next().unwrap_or_default().to_owned();
                requests.lock().unwrap().push(line);
                let body = format!(r#"[{{"type":"file","oid":"x","size":4,"path":"{name}"}}]"#);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
            });
        }
    });
    format!("http://{addr}")
}

fn client(endpoint: &str, cache_dir: &Path) -> HFClient {
    HFClient::builder()
        .endpoint(endpoint)
        .cache_dir(cache_dir)
        .build()
        .expect("client construction is offline")
}

fn assert_rejected(result: HFResult<std::path::PathBuf>, name: &str) {
    match result {
        Err(HFError::InvalidParameter(message)) => {
            assert!(message.contains("confined"), "{name:?}: {message}");
        }
        other => panic!("{name:?} was not rejected: {other:?}"),
    }
}

/// Only the tree listing may be requested, and nothing may be written beside the
/// cache or local directory.
fn assert_contained(root: &Path, requests: &Mutex<Vec<String>>) {
    let requests = requests.lock().unwrap();
    assert!(!requests.is_empty());
    assert!(
        requests.iter().all(|line| line.contains("/tree/")),
        "{requests:?}"
    );
    let mut entries: Vec<_> = std::fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    entries.sort();
    assert_eq!(entries, ["cache", "out"]);
    assert!(!root.join("escape.bin").exists());
}

#[tokio::test]
async fn snapshot_download_rejects_listed_names_that_escape() {
    for name in HOSTILE_NAMES {
        for into_local_dir in [false, true] {
            let requests = Arc::new(Mutex::new(Vec::new()));
            let endpoint = start_tree_server(name, Arc::clone(&requests)).await;
            let root = tempfile::tempdir().unwrap();
            // Nest both targets so a name climbing out of them lands in `root`.
            let cache_dir = root.path().join("cache").join("hub");
            let local_dir = root.path().join("out").join("model");
            std::fs::create_dir_all(&cache_dir).unwrap();
            std::fs::create_dir_all(&local_dir).unwrap();

            let repo = client(&endpoint, &cache_dir).model("owner", "repo");
            let download = repo.snapshot_download().revision(COMMIT);
            let result = if into_local_dir {
                download.local_dir(local_dir.clone()).send().await
            } else {
                download.send().await
            };

            assert_rejected(result, name);
            assert_contained(root.path(), &requests);
        }
    }
}

#[tokio::test]
async fn download_file_rejects_names_that_escape_before_any_request() {
    for name in HOSTILE_NAMES {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let endpoint = start_tree_server(name, Arc::clone(&requests)).await;
        let root = tempfile::tempdir().unwrap();
        let cache_dir = root.path().join("cache").join("hub");
        let local_dir = root.path().join("out").join("model");
        std::fs::create_dir_all(&cache_dir).unwrap();
        std::fs::create_dir_all(&local_dir).unwrap();
        let repo = client(&endpoint, &cache_dir).model("owner", "repo");

        let cached = repo.download_file().filename(name).revision(COMMIT).send();
        assert_rejected(cached.await, name);
        let local = repo
            .download_file()
            .filename(name)
            .revision(COMMIT)
            .local_dir(local_dir)
            .send();
        assert_rejected(local.await, name);

        assert!(requests.lock().unwrap().is_empty());
        assert!(!root.path().join("escape.bin").exists());
    }
}

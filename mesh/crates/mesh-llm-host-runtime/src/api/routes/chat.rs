use super::super::{MeshApi, http::respond_error};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Whether the management caller behind a forwarded inference request passed
/// the trusted-local check (loopback peer + local `Host`/`Origin`).
///
/// The console forwards inference to the OpenAI port over a fresh loopback
/// socket, so without this the exchange hook's caller-trust
/// check would see every forwarded caller as local
/// — including remote ones reaching a `--listen-all` console (Loupe #3051).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CallerTrust {
    TrustedLocal,
    Untrusted,
}

impl CallerTrust {
    pub(crate) fn from_trusted_local(trusted: bool) -> Self {
        if trusted {
            Self::TrustedLocal
        } else {
            Self::Untrusted
        }
    }
}

/// Upper bound on the upstream response head we buffer before relaying.
const MAX_UPSTREAM_HEAD_BYTES: usize = 64 * 1024;

/// Headers forced onto every response the console relays from the inference
/// port. The body is produced by whichever mesh peer served the request, so
/// it must never be able to execute on the management console's origin
/// (Loupe #2834): `sandbox` gives it an opaque origin even if it is HTML, and
/// `nosniff` stops browsers from guessing a script/HTML type. Upstream
/// `X-Content-Type-Options` lines are removed first so ours is the only one.
const RELAY_SECURITY_HEADERS: &str =
    "X-Content-Type-Options: nosniff\r\nContent-Security-Policy: sandbox\r\n";

pub(super) fn is_inference_passthrough(path: &str) -> bool {
    path.starts_with("/v1/")
        || path == "/models"
        || path.starts_with("/models/")
        || path.starts_with("/mesh/v1/models")
        || path == "/mesh/v1/documents/ocr"
        || path == "/mesh/v1/images/similarity"
        || path == "/mesh/v1/images/depth"
}

pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    method: &str,
    path_only: &str,
    req: &[u8],
    caller: CallerTrust,
) -> anyhow::Result<()> {
    let is_passthrough = is_inference_passthrough(path_only);
    if method == "OPTIONS" && is_passthrough {
        stream
            .write_all(
                b"HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: content-type, authorization\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nContent-Length: 0\r\n\r\n",
            )
            .await?;
        return Ok(());
    }

    // A browser navigation is a GET, and the inference port's model-less
    // fallback would route an arbitrary `/v1/*` GET to whichever peer is first
    // in the target list. Only recognized model discovery routes are legitimate GETs here.
    let allowed_get =
        method == "GET" && crate::network::openai::model_discovery::is_request(method, path_only);
    if method != "POST" && !allowed_get {
        return respond_error(stream, 405, "Method Not Allowed").await;
    }

    let alias_target = if is_passthrough {
        None
    } else {
        match console_alias_target(path_only) {
            Some(target) => Some(target),
            None => return respond_error(stream, 404, "Not found").await,
        }
    };
    let forwarded = match alias_target {
        None => req.to_vec(),
        Some(target) => match rewrite_request_target(req, path_only, target) {
            Some(rewritten) => rewritten,
            None => return respond_error(stream, 400, "Bad request line").await,
        },
    };

    let port = state.inner.lock().await.api_port;

    let target = format!("127.0.0.1:{port}");
    match TcpStream::connect(&target).await {
        Ok(mut upstream) => {
            // A remote caller must stay remote across the loopback hop so the
            // exchange hook admits (or refuses) it like any other remote peer.
            let _remote_origin = match caller {
                CallerTrust::TrustedLocal => None,
                CallerTrust::Untrusted => Some(
                    crate::network::management_origin::UntrustedManagementRequest::register(
                        upstream.local_addr()?,
                    )?,
                ),
            };
            upstream.write_all(&forwarded).await?;
            relay_upstream_response(stream, &mut upstream).await?;
        }
        _ => {
            respond_error(stream, 502, "Cannot reach LLM server").await?;
        }
    }
    Ok(())
}

/// Inference route behind a console alias. Only the exact alias paths are
/// accepted, so `/api/chatty` or `/api/chat/x` are not rewritten into odd
/// upstream paths.
fn console_alias_target(path: &str) -> Option<&'static str> {
    match path {
        "/api/chat" => Some("/v1/chat/completions"),
        "/api/responses" => Some("/v1/responses"),
        _ => None,
    }
}

/// Replace the path of the request target with `target`, keeping the query
/// string and every byte after the request line unchanged. The body is never
/// decoded, so non-UTF-8 bytes and `Content-Length` stay consistent.
///
/// Returns `None` unless the request line is `METHOD SP PATH[?QUERY] SP
/// VERSION` with `PATH` equal to `alias`.
fn rewrite_request_target(req: &[u8], alias: &str, target: &str) -> Option<Vec<u8>> {
    let line_end = req.windows(2).position(|w| w == b"\r\n")?;
    let (line, rest) = req.split_at(line_end);
    let mut parts = line.split(|&b| b == b' ');
    let (method, request_target, version) = (parts.next()?, parts.next()?, parts.next()?);
    if parts.next().is_some() {
        return None;
    }
    let query_start = request_target
        .iter()
        .position(|&b| b == b'?')
        .unwrap_or(request_target.len());
    let (path, query) = request_target.split_at(query_start);
    if path != alias.as_bytes() {
        return None;
    }
    let mut out = Vec::with_capacity(req.len() + target.len());
    out.extend_from_slice(method);
    out.push(b' ');
    out.extend_from_slice(target.as_bytes());
    out.extend_from_slice(query);
    out.push(b' ');
    out.extend_from_slice(version);
    out.extend_from_slice(rest);
    Some(out)
}

/// Relay the upstream response, splicing [`RELAY_SECURITY_HEADERS`] into
/// every response head, then stream the rest bidirectionally (request bodies
/// may still be in flight for streaming clients).
///
/// Interim `1xx` heads (such as `103 Early Hints`) are relayed and parsing
/// continues, so the final response head also carries the headers.
///
/// The bytes come from an untrusted peer, so the relay fails closed: if it
/// cannot frame a head exactly as a browser would (see
/// [`secure_response_head`]), or upstream closes before a complete head, the
/// client gets a 502 and none of the upstream bytes.
async fn relay_upstream_response(
    client: &mut TcpStream,
    upstream: &mut TcpStream,
) -> anyhow::Result<()> {
    let mut buf = Vec::with_capacity(1024);
    loop {
        let secured = read_response_head(upstream, &mut buf)
            .await?
            .and_then(|head_len| secure_response_head(&buf[..head_len]).map(|s| (head_len, s)));
        let Some((head_len, secured)) = secured else {
            return respond_error(client, 502, "Invalid upstream response").await;
        };
        buf.drain(..head_len);
        client.write_all(&secured.head).await?;
        if !secured.interim {
            break;
        }
    }
    client.write_all(&buf).await?;
    tokio::io::copy_bidirectional(client, upstream).await?;
    Ok(())
}

/// Read from `upstream` until `buf` holds a complete response head. Returns
/// the head length including the terminating blank line, or `None` if
/// upstream closed first or the head exceeds [`MAX_UPSTREAM_HEAD_BYTES`].
async fn read_response_head(
    upstream: &mut TcpStream,
    buf: &mut Vec<u8>,
) -> anyhow::Result<Option<usize>> {
    loop {
        if let Some(end) = header_end(buf) {
            let head_len = end + 2;
            return Ok((head_len <= MAX_UPSTREAM_HEAD_BYTES).then_some(head_len));
        }
        if buf.len() >= MAX_UPSTREAM_HEAD_BYTES {
            return Ok(None);
        }
        let mut chunk = [0u8; 4096];
        let n = upstream.read(&mut chunk).await?;
        if n == 0 {
            return Ok(None);
        }
        buf.extend_from_slice(&chunk[..n]);
    }
}

/// A response head ready to send to the client.
struct SecuredHead {
    head: Vec<u8>,
    /// An interim `1xx` response; another head follows.
    interim: bool,
}

/// Validate one upstream response head (ending in `\r\n\r\n`) and return it
/// with [`RELAY_SECURITY_HEADERS`] added.
///
/// Returns `None` for any head where a browser could see a different
/// framing than the relay: a bare CR or LF, an obsolete folded header line,
/// or a status line other than `HTTP/1.0` or `HTTP/1.1`, one space and three
/// digits. Upstream `X-Content-Type-Options` lines are dropped, because a
/// browser only checks the first value of that header.
fn secure_response_head(head: &[u8]) -> Option<SecuredHead> {
    let lines_end = head.len().checked_sub(4)?;
    if !has_only_crlf_line_endings(head) {
        return None;
    }
    let mut lines = head[..lines_end]
        .split(|&b| b == b'\n')
        .map(|line| line.strip_suffix(b"\r").unwrap_or(line));
    let status_line = lines.next()?;
    let status = parse_status_code(status_line)?;
    let mut out = Vec::with_capacity(head.len() + RELAY_SECURITY_HEADERS.len());
    out.extend_from_slice(status_line);
    out.extend_from_slice(b"\r\n");
    for line in lines {
        if line.starts_with(b" ") || line.starts_with(b"\t") {
            return None;
        }
        if !is_header_named(line, "x-content-type-options") {
            out.extend_from_slice(line);
            out.extend_from_slice(b"\r\n");
        }
    }
    out.extend_from_slice(RELAY_SECURITY_HEADERS.as_bytes());
    out.extend_from_slice(b"\r\n");
    Some(SecuredHead {
        head: out,
        // `101 Switching Protocols` ends HTTP framing, so it counts as final.
        interim: (100..200).contains(&status) && status != 101,
    })
}

/// Whether every CR is followed by LF and every LF follows a CR.
fn has_only_crlf_line_endings(head: &[u8]) -> bool {
    head.iter().enumerate().all(|(i, &b)| match b {
        b'\r' => head.get(i + 1) == Some(&b'\n'),
        b'\n' => i > 0 && head[i - 1] == b'\r',
        _ => true,
    })
}

/// Status code of a strict `HTTP/1.x SP 3DIGIT [SP reason]` status line.
fn parse_status_code(line: &[u8]) -> Option<u16> {
    let rest = line
        .strip_prefix(b"HTTP/1.1 ")
        .or_else(|| line.strip_prefix(b"HTTP/1.0 "))?;
    let (code, after) = rest.split_at_checked(3)?;
    if !code.iter().all(u8::is_ascii_digit) || !(after.is_empty() || after.starts_with(b" ")) {
        return None;
    }
    std::str::from_utf8(code).ok()?.parse().ok()
}

/// Whether the header line's field name is `name`, ignoring ASCII case and
/// whitespace around the name.
fn is_header_named(line: &[u8], name: &str) -> bool {
    line.iter().position(|&b| b == b':').is_some_and(|colon| {
        line[..colon]
            .trim_ascii()
            .eq_ignore_ascii_case(name.as_bytes())
    })
}

/// Offset of the start of the blank line terminating the response head
/// (the position where an extra header line can be inserted).
fn header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_end_points_at_blank_line() {
        let head = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<html>";
        let end = header_end(head).unwrap();
        assert_eq!(
            &head[..end],
            b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
        );
        assert_eq!(header_end(b"HTTP/1.1 200 OK\r\nX: y\r\n"), None);
    }

    /// Run [`relay_upstream_response`] against an upstream that writes
    /// `response` and closes, returning what the browser received.
    async fn relay_fixed_response(response: Vec<u8>) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let upstream_task = tokio::spawn(async move {
            let (mut s, _) = listener.accept().await.unwrap();
            // The relay may stop reading early (oversized head), so a failed
            // write is expected there.
            let _ = s.write_all(&response).await;
            let _ = s.shutdown().await;
        });
        let client_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let client_addr = client_listener.local_addr().unwrap();
        let mut browser = TcpStream::connect(client_addr).await.unwrap();
        let (mut client, _) = client_listener.accept().await.unwrap();
        let mut upstream = TcpStream::connect(addr).await.unwrap();
        browser.shutdown().await.unwrap();
        relay_upstream_response(&mut client, &mut upstream)
            .await
            .unwrap();
        drop(client);
        drop(upstream);
        let mut out = Vec::new();
        browser.read_to_end(&mut out).await.unwrap();
        upstream_task.await.unwrap();
        String::from_utf8(out).unwrap()
    }

    #[tokio::test]
    async fn relay_forces_sandbox_and_nosniff_on_upstream_html() {
        let body = "<script>alert(1)</script>";
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let text = relay_fixed_response(resp.into_bytes()).await;
        assert!(text.starts_with("HTTP/1.1 200 OK\r\n"), "{text}");
        assert!(
            text.contains("X-Content-Type-Options: nosniff\r\n"),
            "{text}"
        );
        assert!(
            text.contains("Content-Security-Policy: sandbox\r\n"),
            "{text}"
        );
        assert!(text.ends_with(&format!("\r\n\r\n{body}")), "{text}");
    }

    #[tokio::test]
    async fn relay_secures_final_head_after_interim_responses() {
        let body = "<script>alert(1)</script>";
        let resp = format!(
            "HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 103 Early Hints\r\nLink: </x.css>; rel=preload\r\n\r\nHTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let text = relay_fixed_response(resp.into_bytes()).await;
        let final_start = text
            .find("HTTP/1.1 200 OK\r\n")
            .expect("final head relayed");
        let (interim, final_response) = text.split_at(final_start);
        assert!(interim.starts_with("HTTP/1.1 100 Continue\r\n"), "{text}");
        assert!(interim.contains("HTTP/1.1 103 Early Hints\r\n"), "{text}");
        let final_head = &final_response[..final_response.find("\r\n\r\n").unwrap()];
        assert!(
            final_head.contains("X-Content-Type-Options: nosniff"),
            "{text}"
        );
        assert!(
            final_head.contains("Content-Security-Policy: sandbox"),
            "{text}"
        );
        assert!(
            final_response.ends_with(&format!("\r\n\r\n{body}")),
            "{text}"
        );
    }

    /// Assert the relay answered with its own 502 and passed none of the
    /// upstream response through.
    fn assert_relay_rejected(text: &str) {
        assert!(text.starts_with("HTTP/1.1 502 Bad Gateway\r\n"), "{text}");
        assert_eq!(text.matches("HTTP/1.").count(), 1, "{text}");
        assert!(!text.contains("<script>"), "{text}");
    }

    #[tokio::test]
    async fn relay_rejects_bare_lf_heads() {
        // A browser ends this head at `\n\n` and runs the script; the relay
        // would only find `\r\n\r\n` inside the body.
        let text = relay_fixed_response(
            b"HTTP/1.1 200 OK\nContent-Type: text/html\n\n<script>alert(1)</script>\r\n\r\n"
                .to_vec(),
        )
        .await;
        assert_relay_rejected(&text);
    }

    #[tokio::test]
    async fn relay_rejects_upstream_close_before_complete_head() {
        let text = relay_fixed_response(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n<script>alert(1)</script>".to_vec(),
        )
        .await;
        assert_relay_rejected(&text);
    }

    #[tokio::test]
    async fn relay_rejects_oversized_head() {
        let mut resp = b"HTTP/1.1 200 OK\r\nX-Pad: ".to_vec();
        resp.resize(resp.len() + MAX_UPSTREAM_HEAD_BYTES, b'a');
        resp.extend_from_slice(b"\r\n\r\n<script>alert(1)</script>");
        let text = relay_fixed_response(resp).await;
        assert_relay_rejected(&text);
    }

    #[tokio::test]
    async fn relay_rejects_lenient_status_line_after_interim() {
        // Browsers read the second head as another `100` and render the
        // third one, so the relay must not treat the second as final.
        let text = relay_fixed_response(
            b"HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1  100 Continue\r\n\r\nHTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<script>alert(1)</script>"
                .to_vec(),
        )
        .await;
        let reject = text.find("HTTP/1.1 502").expect("relay rejects");
        assert!(
            text[..reject].starts_with("HTTP/1.1 100 Continue\r\n"),
            "{text}"
        );
        assert_relay_rejected(&text[reject..]);
    }

    #[tokio::test]
    async fn relay_replaces_upstream_nosniff_opt_out() {
        let text = relay_fixed_response(
            b"HTTP/1.1 200 OK\r\nx-content-type-options : none\r\nContent-Type: text/html\r\n\r\nhi"
                .to_vec(),
        )
        .await;
        let head = &text[..text.find("\r\n\r\n").unwrap()].to_ascii_lowercase();
        assert_eq!(head.matches("x-content-type-options").count(), 1, "{text}");
        assert!(head.contains("x-content-type-options: nosniff"), "{text}");
        assert!(text.ends_with("\r\n\r\nhi"), "{text}");
    }

    #[test]
    fn secure_response_head_classifies_status() {
        let interim = |head: &[u8]| secure_response_head(head).map(|s| s.interim);
        assert_eq!(interim(b"HTTP/1.1 100 Continue\r\n\r\n"), Some(true));
        assert_eq!(interim(b"HTTP/1.1 103 Early Hints\r\n\r\n"), Some(true));
        assert_eq!(interim(b"HTTP/1.0 103\r\n\r\n"), Some(true));
        assert_eq!(
            interim(b"HTTP/1.1 101 Switching Protocols\r\n\r\n"),
            Some(false)
        );
        assert_eq!(interim(b"HTTP/1.1 200 OK\r\nX: y\r\n\r\n"), Some(false));
        for head in [
            &b"HTTP/1.1  100 Continue\r\n\r\n"[..],
            b"HTTP/1.10 100 Continue\r\n\r\n",
            b"HTTP/1.1 100Continue\r\n\r\n",
            b"HTTP/2 100 Continue\r\n\r\n",
            b"HTTP/1.1 1xx Bogus\r\n\r\n",
            b"HTTP/1.1 20\r\n\r\n",
            b"garbage\r\n\r\n",
            b"\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nX: a\rb\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nX: a\r\n b\r\n\r\n",
            b"HTTP/1.1 200 OK\r\nX: a\r\n\tb\r\n\r\n",
        ] {
            assert!(
                secure_response_head(head).is_none(),
                "{}",
                String::from_utf8_lossy(head)
            );
        }
    }

    #[test]
    fn secure_response_head_keeps_other_headers() {
        let secured = secure_response_head(
            b"HTTP/1.1 200 OK\r\nContent-Security-Policy: default-src 'none'\r\nX-Content-Type-Options: nosniff\r\nContent-Length: 2\r\n\r\n",
        )
        .unwrap();
        assert_eq!(
            secured.head,
            b"HTTP/1.1 200 OK\r\nContent-Security-Policy: default-src 'none'\r\nContent-Length: 2\r\nX-Content-Type-Options: nosniff\r\nContent-Security-Policy: sandbox\r\n\r\n"
        );
    }

    #[test]
    fn console_alias_requires_exact_path() {
        assert_eq!(
            console_alias_target("/api/chat"),
            Some("/v1/chat/completions")
        );
        assert_eq!(
            console_alias_target("/api/responses"),
            Some("/v1/responses")
        );
        for path in [
            "/api/chatty",
            "/api/chat/",
            "/api/chat/completions",
            "/api/responsesx",
            "/api/responses/1",
        ] {
            assert_eq!(console_alias_target(path), None, "{path}");
        }
    }

    #[test]
    fn rewrite_changes_only_the_request_line_path() {
        let mut req = b"POST /api/chat?trace=1 HTTP/1.1\r\nHost: localhost\r\nReferer: http://localhost/api/chat\r\nContent-Length: 6\r\n\r\n".to_vec();
        req.extend_from_slice(b"\x00\xff\x80/api");
        let out = rewrite_request_target(&req, "/api/chat", "/v1/chat/completions").unwrap();
        let mut expected = b"POST /v1/chat/completions?trace=1 HTTP/1.1\r\nHost: localhost\r\nReferer: http://localhost/api/chat\r\nContent-Length: 6\r\n\r\n".to_vec();
        expected.extend_from_slice(b"\x00\xff\x80/api");
        assert_eq!(out, expected);
    }

    #[test]
    fn rewrite_rejects_mismatched_or_malformed_request_lines() {
        let alias = "/api/chat";
        let target = "/v1/chat/completions";
        for req in [
            &b"POST /api/chatty HTTP/1.1\r\n\r\n"[..],
            b"POST /api/chat/x HTTP/1.1\r\n\r\n",
            b"POST /x/api/chat HTTP/1.1\r\n\r\n",
            b"POST /api/chat HTTP/1.1 extra\r\n\r\n",
            b"POST /api/chat\r\n\r\n",
            b"POST /api/chat HTTP/1.1",
        ] {
            assert_eq!(
                rewrite_request_target(req, alias, target),
                None,
                "{}",
                String::from_utf8_lossy(req)
            );
        }
    }
}

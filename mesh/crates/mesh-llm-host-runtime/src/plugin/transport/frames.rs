//! Length-prefixed envelope frames on the plugin control stream.
//!
//! A frame read is two `read_exact` calls, so it is not cancel-safe: if the
//! future is dropped part way through a frame, the bytes it already took are
//! gone and the stream loses its place. The connection loop therefore keeps
//! one read future alive for the life of the connection and never races it
//! against other work.
use super::proto;
use anyhow::{Result, anyhow, bail};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

// An "external relay" style plugin (one whose real inference traffic
// bypasses this IPC channel entirely, going through its own HTTP proxy
// instead — see `openai_http(...).managed_by_plugin(false)` plugins)
// generates no application-level IPC traffic of its own. Its reply to the
// supervisor's periodic `HealthRequest` (see
// `health::HEALTH_CHECK_INTERVAL_SECS`) is the *only* guaranteed inbound
// traffic such a plugin's connection ever sees, so this timeout must
// comfortably outlast that cadence — a timeout shorter than the health-check
// interval would disconnect every idle-but-healthy relay plugin as falsely
// idle before its next probe ever arrives. 4x the health interval plus a
// fixed floor covers a slow/jittery tick without staying pinned to an
// arbitrarily large value.
pub(super) const PREFIX_READ_TIMEOUT: Duration =
    Duration::from_secs((super::super::health::HEALTH_CHECK_INTERVAL_SECS * 4) + 30);
pub(super) const BODY_READ_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

pub(super) async fn write_frame<W>(stream: &mut W, envelope: &proto::Envelope) -> Result<()>
where
    W: AsyncWrite + Unpin + ?Sized,
{
    let mut body = Vec::new();
    prost::Message::encode(envelope, &mut body)?;
    stream.write_all(&(body.len() as u32).to_le_bytes()).await?;
    stream.write_all(&body).await?;
    stream.flush().await?;
    Ok(())
}

pub(super) async fn read_frame<R>(
    stream: &mut R,
    prefix_timeout: Duration,
    body_timeout: Duration,
) -> Result<proto::Envelope>
where
    R: AsyncRead + Unpin + ?Sized,
{
    let mut len_buf = [0u8; 4];
    tokio::time::timeout(prefix_timeout, stream.read_exact(&mut len_buf))
        .await
        .map_err(|_| anyhow!("timeout reading plugin frame prefix after {prefix_timeout:?}"))??;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > MAX_FRAME_BYTES {
        bail!("Plugin frame too large");
    }
    let mut body = vec![0u8; len];
    tokio::time::timeout(body_timeout, stream.read_exact(&mut body))
        .await
        .map_err(|_| anyhow!("timeout reading plugin frame body after {body_timeout:?}"))??;
    Ok(prost::Message::decode(body.as_slice())?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_READ_TIMEOUT: Duration = Duration::from_secs(1);

    #[tokio::test(start_paused = true)]
    async fn prefix_and_body_reads_have_timeouts() {
        let (_prefix_writer, mut prefix_reader) = tokio::io::duplex(1);
        let prefix_error = read_frame(&mut prefix_reader, TEST_READ_TIMEOUT, TEST_READ_TIMEOUT)
            .await
            .expect_err("stalled prefix read must time out");
        assert!(
            prefix_error
                .to_string()
                .contains("timeout reading plugin frame prefix")
        );

        let (mut body_writer, mut body_reader) = tokio::io::duplex(8);
        body_writer.write_all(&4u32.to_le_bytes()).await.unwrap();
        let body_error = read_frame(&mut body_reader, TEST_READ_TIMEOUT, TEST_READ_TIMEOUT)
            .await
            .expect_err("stalled body read must time out");
        assert!(
            body_error
                .to_string()
                .contains("timeout reading plugin frame body")
        );
    }

    #[tokio::test]
    async fn oversized_frames_are_rejected_before_allocating() {
        let (mut writer, mut reader) = tokio::io::duplex(8);
        writer
            .write_all(&((MAX_FRAME_BYTES as u32) + 1).to_le_bytes())
            .await
            .unwrap();
        let error = read_frame(&mut reader, TEST_READ_TIMEOUT, TEST_READ_TIMEOUT)
            .await
            .expect_err("an oversized frame must be refused");
        assert!(error.to_string().contains("too large"));
    }
}

//! Reading one length-prefixed plugin frame from a peer's stream.
//!
//! Plugin channel and bulk-transfer streams carry a single frame whose
//! length prefix a peer chooses. The prefix and the body each have to arrive
//! within a deadline, and the body buffer grows only with the bytes that
//! actually arrive.

use anyhow::{Context, Result};
use tokio::io::AsyncReadExt as _;

const PLUGIN_FRAME_PREFIX_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
const PLUGIN_FRAME_BODY_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

pub(super) async fn read_plugin_frame_bytes<R>(reader: &mut R, max_len: usize) -> Result<Vec<u8>>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut len_buf = [0u8; 4];
    tokio::time::timeout(
        PLUGIN_FRAME_PREFIX_READ_TIMEOUT,
        reader.read_exact(&mut len_buf),
    )
    .await
    .map_err(|_| {
        anyhow::anyhow!(
            "timeout reading plugin frame prefix after {PLUGIN_FRAME_PREFIX_READ_TIMEOUT:?}"
        )
    })?
    .context("read plugin frame prefix")?;
    let len = u32::from_le_bytes(len_buf) as usize;
    anyhow::ensure!(len <= max_len, "Plugin frame too large");
    // The buffer grows as the body arrives instead of being sized from the
    // prefix, so a peer that claims a large frame and then stalls holds only
    // as much memory as it actually sent.
    let mut buf = Vec::new();
    tokio::time::timeout(
        PLUGIN_FRAME_BODY_READ_TIMEOUT,
        (&mut *reader).take(len as u64).read_to_end(&mut buf),
    )
    .await
    .map_err(|_| {
        anyhow::anyhow!(
            "timeout reading plugin frame body after {PLUGIN_FRAME_BODY_READ_TIMEOUT:?}"
        )
    })?
    .context("read plugin frame body")?;
    anyhow::ensure!(
        buf.len() == len,
        "plugin frame body ended after {} of {len} bytes",
        buf.len()
    );
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::pin::Pin;
    use std::task::{Context as TaskContext, Poll};
    use tokio::io::{AsyncRead, AsyncWriteExt as _, ReadBuf};

    #[tokio::test(start_paused = true)]
    async fn plugin_frame_prefix_and_body_reads_have_timeouts() {
        let (_prefix_writer, mut prefix_reader) = tokio::io::duplex(1);
        let prefix_error = read_plugin_frame_bytes(&mut prefix_reader, 1024)
            .await
            .expect_err("stalled prefix read must time out");
        assert!(
            prefix_error
                .to_string()
                .contains("timeout reading plugin frame prefix")
        );

        let (mut body_writer, mut body_reader) = tokio::io::duplex(8);
        body_writer.write_all(&4u32.to_le_bytes()).await.unwrap();
        let body_error = read_plugin_frame_bytes(&mut body_reader, 1024)
            .await
            .expect_err("stalled body read must time out");
        assert!(
            body_error
                .to_string()
                .contains("timeout reading plugin frame body")
        );
    }

    /// Serves `data` and then EOF, recording the largest read buffer the
    /// frame reader offered it.
    struct RecordingReader {
        data: Vec<u8>,
        offset: usize,
        largest_read_buffer: usize,
    }

    impl AsyncRead for RecordingReader {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _cx: &mut TaskContext<'_>,
            buf: &mut ReadBuf<'_>,
        ) -> Poll<std::io::Result<()>> {
            self.largest_read_buffer = self.largest_read_buffer.max(buf.remaining());
            let rest = &self.data[self.offset..];
            let n = rest.len().min(buf.remaining());
            buf.put_slice(&rest[..n]);
            self.offset += n;
            Poll::Ready(Ok(()))
        }
    }

    #[tokio::test]
    async fn a_large_claimed_frame_is_not_allocated_before_its_body_arrives() {
        // The peer claims a 64 MB frame and sends only 16 bytes of it.
        let claimed = 64_000_000u32;
        let mut data = claimed.to_le_bytes().to_vec();
        data.extend_from_slice(&[0xAB; 16]);
        let mut reader = RecordingReader {
            data,
            offset: 0,
            largest_read_buffer: 0,
        };

        let error = read_plugin_frame_bytes(&mut reader, claimed as usize)
            .await
            .expect_err("a truncated frame must be rejected");

        assert!(
            error
                .to_string()
                .contains("ended after 16 of 64000000 bytes"),
            "{error}"
        );
        assert!(
            reader.largest_read_buffer <= 64 * 1024,
            "the body buffer must grow with the bytes received, not the claimed \
             length; the reader was offered {} bytes",
            reader.largest_read_buffer
        );
    }

    #[tokio::test]
    async fn a_complete_frame_is_returned_whole() {
        let body = b"plugin frame body".to_vec();
        let mut data = (body.len() as u32).to_le_bytes().to_vec();
        data.extend_from_slice(&body);
        let mut reader = RecordingReader {
            data,
            offset: 0,
            largest_read_buffer: 0,
        };
        assert_eq!(
            read_plugin_frame_bytes(&mut reader, 1024).await.unwrap(),
            body
        );
    }

    #[tokio::test]
    async fn a_frame_over_the_limit_is_rejected_from_its_prefix() {
        let mut reader = RecordingReader {
            data: 1025u32.to_le_bytes().to_vec(),
            offset: 0,
            largest_read_buffer: 0,
        };
        let error = read_plugin_frame_bytes(&mut reader, 1024)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("too large"), "{error}");
    }
}

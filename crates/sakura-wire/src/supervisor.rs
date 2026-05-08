//! WorkerSupervisor: spawn `sakura-worker` subprocesses, harvest their listen URI
//! from stdout, and shut them down cleanly.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum SupervisorError {
    #[error("spawn: {0}")]
    Spawn(#[from] std::io::Error),
    #[error("worker exited before reporting URI")]
    EarlyExit,
    #[error("did not receive listen URI within {0:?}")]
    Timeout(Duration),
    #[error("invalid listen URI: {0}")]
    BadUri(String),
}

pub struct WorkerHandle {
    pub uri: String,
    pub cert_der: Vec<u8>,
    child: Mutex<Child>,
}

impl WorkerHandle {
    pub fn shutdown(&self, timeout: Duration) {
        let _ = self.child.lock().unwrap().kill();
        let mut waited = Duration::ZERO;
        let step = Duration::from_millis(50);
        while waited < timeout {
            if self
                .child
                .lock()
                .unwrap()
                .try_wait()
                .ok()
                .flatten()
                .is_some()
            {
                return;
            }
            std::thread::sleep(step);
            waited += step;
        }
        let _ = self.child.lock().unwrap().wait();
    }
}

/// Spawn the worker, read lines from its stdout until the magic line
/// `SAKURA_WORKER_LISTENING <uri> <cert_hex>` appears, then return its handle.
pub fn spawn_worker(
    cmd: &[String],
    extra_env: HashMap<String, String>,
    startup_timeout: Duration,
) -> Result<WorkerHandle, SupervisorError> {
    if cmd.is_empty() {
        return Err(SupervisorError::BadUri("empty cmd".into()));
    }
    let mut command = Command::new(&cmd[0]);
    command
        .args(&cmd[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    for (k, v) in extra_env {
        command.env(k, v);
    }
    let mut child = command.spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| SupervisorError::Spawn(std::io::Error::other("no stdout")))?;

    let (tx, rx) = std::sync::mpsc::channel::<Result<(String, Vec<u8>), SupervisorError>>();
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            match line {
                Ok(line) if line.starts_with("SAKURA_WORKER_LISTENING ") => {
                    let parts: Vec<&str> = line.splitn(3, ' ').collect();
                    if parts.len() != 3 {
                        let _ = tx.send(Err(SupervisorError::BadUri(line)));
                        return;
                    }
                    let uri = parts[1].to_string();
                    let cert_hex = parts[2];
                    match decode_hex(cert_hex) {
                        Ok(cert) => {
                            let _ = tx.send(Ok((uri, cert)));
                        }
                        Err(e) => {
                            let _ =
                                tx.send(Err(SupervisorError::BadUri(format!("bad cert hex: {e}"))));
                        }
                    }
                    return;
                }
                Ok(_) => continue,
                Err(_) => {
                    let _ = tx.send(Err(SupervisorError::EarlyExit));
                    return;
                }
            }
        }
        let _ = tx.send(Err(SupervisorError::EarlyExit));
    });

    match rx.recv_timeout(startup_timeout) {
        Ok(Ok((uri, cert_der))) => Ok(WorkerHandle {
            uri,
            cert_der,
            child: Mutex::new(child),
        }),
        Ok(Err(e)) => {
            let _ = child.kill();
            Err(e)
        }
        Err(_) => {
            let _ = child.kill();
            Err(SupervisorError::Timeout(startup_timeout))
        }
    }
}

fn decode_hex(s: &str) -> Result<Vec<u8>, String> {
    if !s.len().is_multiple_of(2) {
        return Err("odd-length hex".into());
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    for i in (0..s.len()).step_by(2) {
        let byte =
            u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| format!("bad hex byte: {e}"))?;
        out.push(byte);
    }
    Ok(out)
}

pub fn encode_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

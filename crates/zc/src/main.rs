mod artifacts;
mod cli;
mod models;

use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use std::fs;
use std::process::Stdio;
use tokio::process::Command;
use tokio::time::Duration;

use cli::{Cli, Commands};
use models::{ExecutionPlan, ExecutionResult};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Execute { plan, json } => {
            let plan_path = plan;
            let plan_content = fs::read_to_string(&plan_path)
                .with_context(|| format!("Failed to read plan file: {:?}", plan_path))?;
            let execution_plan: ExecutionPlan = serde_json::from_str(&plan_content)
                .context("Failed to parse execution plan JSON")?;

            let started_at = Utc::now();
            let job_id = format!("job-{}", started_at.format("%Y%m%d-%H%M%S-%f"));

            // Determine artifact directory
            let artifact_root = "zakuro-artifacts";
            let artifact_dir_path = artifacts::prepare_artifact_dir(artifact_root, &job_id)?;

            // Save the plan to the artifact directory as required by the plugin
            fs::copy(&plan_path, artifact_dir_path.join("plan.json"))
                .context("Failed to copy plan.json to artifact directory")?;

            let mut status = "failed".to_string();
            let mut stdout_data = String::new();
            let mut stderr_data = String::new();
            let mut exit_code = None;
            let mut error_message = None;

            if execution_plan.command.is_empty() {
                error_message = Some("Empty command in execution plan".to_string());
            } else {
                let mut cmd = Command::new(&execution_plan.command[0]);
                if execution_plan.command.len() > 1 {
                    cmd.args(&execution_plan.command[1..]);
                }

                cmd.stdout(Stdio::piped());
                cmd.stderr(Stdio::piped());

                if let Some(ref work_dir) = execution_plan.working_dir {
                    if work_dir.is_empty() || work_dir.starts_with('/') || work_dir.contains("..") {
                        return Err(anyhow::anyhow!("Invalid or unsafe working directory"));
                    }
                    let workspace = artifact_dir_path.join("workspace");
                    let full_work_dir = workspace.join(work_dir);
                    fs::create_dir_all(&full_work_dir)?;
                    cmd.current_dir(full_work_dir);
                } else {
                    cmd.current_dir(artifact_dir_path.join("workspace"));
                }

                if let Some(ref env_vars) = execution_plan.env {
                    cmd.envs(env_vars);
                }

                let timeout_secs = execution_plan.resource_limits.timeout_seconds;
                if timeout_secs <= 0 {
                    return Err(anyhow::anyhow!("Timeout must be positive"));
                }
                let timeout_duration = Duration::from_secs(timeout_secs as u64);

                let mut child = cmd.spawn().context("Failed to spawn command")?;

                // Collect the output manually
                let mut stdout = child.stdout.take().context("No stdout")?;
                let mut stderr = child.stderr.take().context("No stderr")?;

                let mut stdout_data_raw = Vec::new();
                let mut stderr_data_raw = Vec::new();

                let wait_handle = child.wait();
                let stdout_handle = tokio::io::AsyncReadExt::read_to_end(&mut stdout, &mut stdout_data_raw);
                let stderr_handle = tokio::io::AsyncReadExt::read_to_end(&mut stderr, &mut stderr_data_raw);

                tokio::select! {
                    res = wait_handle => {
                        let exit_status = res?;
                        let _ = stdout_handle.await;
                        let _ = stderr_handle.await;
                        stdout_data = String::from_utf8_lossy(&stdout_data_raw).to_string();
                        stderr_data = String::from_utf8_lossy(&stderr_data_raw).to_string();
                        exit_code = exit_status.code();
                        status = if exit_status.success() {
                            "succeeded".to_string()
                        } else {
                            "failed".to_string()
                        };
                    }
                    _ = tokio::time::sleep(timeout_duration) => {
                        let _ = child.kill().await;
                        status = "timed_out".to_string();
                        stderr_data = format!(
                            "Job timed out after {} seconds",
                            timeout_secs
                        );
                        error_message = Some(stderr_data.clone());
                    }
                }
            }

            let finished_at = Utc::now();
            let duration_ms = (finished_at - started_at).num_milliseconds();

            // Persistence
            artifacts::write_artifact(&artifact_dir_path, "stdout.txt", &stdout_data)?;
            artifacts::write_artifact(&artifact_dir_path, "stderr.txt", &stderr_data)?;

            let result = ExecutionResult {
                job_id,
                job_name: execution_plan.job_name.clone(),
                backend: "zakuro".to_string(),
                status,
                stdout: stdout_data,
                stderr: stderr_data,
                exit_code,
                duration_ms,
                artifact_dir: artifact_dir_path.to_string_lossy().to_string(),
                started_at,
                finished_at,
                error_message,
            };

            let result_json = serde_json::to_string_pretty(&result)?;
            artifacts::write_artifact(&artifact_dir_path, "result.json", &result_json)?;

            if json {
                println!("{}", result_json);
            } else {
                println!("Job Status: {}", result.status);
                if let Some(code) = result.exit_code {
                    println!("Exit Code: {}", code);
                }
                if !result.stdout.is_empty() {
                    println!("Stdout:\n{}", result.stdout);
                }
                if !result.stderr.is_empty() {
                    println!("Stderr:\n{}", result.stderr);
                }
                if let Some(ref err) = result.error_message {
                    println!("Error: {}", err);
                }
            }

            // Hardened cleanup: Explicitly remove the artifact directory and verify its removal
            if artifact_dir_path.exists() {
                fs::remove_dir_all(&artifact_dir_path).context("Critical: Failed to clean up artifact directory")?;
                if artifact_dir_path.exists() {
                    return Err(anyhow::anyhow!("Critical: Workspace cleanup failed audit"));
                }
            }
        }
    }

    Ok(())
}

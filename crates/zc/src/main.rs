mod artifacts;
mod cli;
mod models;

use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use std::fs;
use std::process::Stdio;
use tokio::process::Command;
use tokio::time::{timeout, Duration};

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
                    let workspace = artifact_dir_path.join("workspace");
                    let full_work_dir = workspace.join(work_dir);
                    // Ensure the directory exists or at least point to it
                    let _ = fs::create_dir_all(&full_work_dir);
                    cmd.current_dir(full_work_dir);
                } else {
                    cmd.current_dir(artifact_dir_path.join("workspace"));
                }

                if let Some(ref env_vars) = execution_plan.env {
                    cmd.envs(env_vars);
                }

                let timeout_duration = Duration::from_secs(execution_plan.resource_limits.timeout_seconds as u64);

                match timeout(timeout_duration, cmd.output()).await {
                    Ok(Ok(output)) => {
                        stdout_data = String::from_utf8_lossy(&output.stdout).to_string();
                        stderr_data = String::from_utf8_lossy(&output.stderr).to_string();
                        exit_code = output.status.code();
                        status = if output.status.success() {
                            "succeeded".to_string()
                        } else {
                            "failed".to_string()
                        };
                    }
                    Ok(Err(e)) => {
                        status = "failed".to_string();
                        stderr_data = format!("Failed to execute command: {}", e);
                        error_message = Some(e.to_string());
                    }
                    Err(_) => {
                        status = "timed_out".to_string();
                        stderr_data = format!(
                            "Job timed out after {} seconds",
                            execution_plan.resource_limits.timeout_seconds
                        );
                        error_message = Some(stderr_data.clone());
                        // Note: Child process cleanup happens when cmd (the Child handle) is dropped, 
                        // but cmd.output() consumes it. If timeout occurs, we might need a more complex
                        // handle to kill it explicitly if it's a persistent process.
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
        }
    }

    Ok(())
}

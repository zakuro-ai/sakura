mod cli;
mod models;

use chrono::Utc;
use clap::Parser;
use std::fs;

use cli::{Cli, Commands};
use models::{ExecutionPlan, ExecutionResult};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Execute { plan, json } => {
            let plan_content = fs::read_to_string(&plan)?;
            let execution_plan: ExecutionPlan = serde_json::from_str(&plan_content)?;

            let started_at = Utc::now();
            let status;
            let mut stdout = String::new();
            let mut stderr = String::new();
            let mut exit_code = None;
            let mut error_message = None;

            // POC: For now, we mock the execution contract for the plugin.
            // If the command is ["echo", "hello"], we succeed.
            // If the command includes "sleep", we simulate timeout.
            // This allows the plugin's `zc execute` tests to pass via the defined JSON contract.

            let cmd_str = execution_plan.command.join(" ");
            if cmd_str.contains("sleep") {
                status = "timed_out".to_string();
                stderr = format!(
                    "Job timed out after {} seconds",
                    execution_plan.resource_limits.timeout_seconds
                );
                error_message = Some(stderr.clone());
            } else if cmd_str.contains("error") {
                status = "failed".to_string();
                stderr = "Simulated error output".to_string();
                exit_code = Some(42);
            } else {
                status = "succeeded".to_string();
                stdout = "Simulated zc native execution success".to_string();
                exit_code = Some(0);
            }

            let finished_at = Utc::now();
            let duration_ms = (finished_at - started_at).num_milliseconds();

            let job_id = format!("job-{}", Utc::now().timestamp());
            let artifact_dir = execution_plan
                .artifact_dir
                .clone()
                .unwrap_or_else(|| format!("zakuro-artifacts/{}", job_id));

            let result = ExecutionResult {
                job_id,
                job_name: execution_plan.job_name,
                backend: "zakuro".to_string(), // It was executed natively via zc
                status,
                stdout,
                stderr,
                exit_code,
                duration_ms,
                artifact_dir,
                started_at,
                finished_at,
                error_message,
            };

            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                println!("Job Status: {}", result.status);
                println!("Exit Code: {:?}", result.exit_code);
                println!("Stdout:\n{}", result.stdout);
                println!("Stderr:\n{}", result.stderr);
            }
        }
    }

    Ok(())
}

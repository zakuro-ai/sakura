use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub cpu_count: f64,
    pub memory_mb: i32,
    pub timeout_seconds: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub job_name: String,
    pub backend: String,
    pub image: String,
    pub command: Vec<String>,
    pub working_dir: Option<String>,
    pub repo_url: Option<String>,
    pub env: Option<HashMap<String, String>>,
    pub resource_limits: ResourceLimits,
    pub artifact_dir: Option<String>,
    pub network_enabled: bool,
    pub created_by: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub job_id: String,
    pub job_name: String,
    pub backend: String,
    pub status: String,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
    pub duration_ms: i64,
    pub artifact_dir: String,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub error_message: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_execution_plan_deserialization() {
        let json = r#"{
            "job_name": "test-job",
            "backend": "zakuro",
            "image": "python:3.11-slim",
            "command": ["echo", "hello"],
            "network_enabled": false,
            "resource_limits": {
                "cpu_count": 1.0,
                "memory_mb": 512,
                "timeout_seconds": 30
            }
        }"#;
        let plan: ExecutionPlan = serde_json::from_str(json).unwrap();
        assert_eq!(plan.job_name, "test-job");
        assert_eq!(plan.command, vec!["echo", "hello"]);
        assert!(!plan.network_enabled);
    }
}

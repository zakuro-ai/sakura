use std::fs;
use std::process::Command;

#[test]
fn test_zc_execute_echo_success() {
    let plan_json = r#"{
        "job_name": "test-echo",
        "backend": "zakuro",
        "image": "python:3.11-slim",
        "command": ["echo", "hello"],
        "network_enabled": false,
        "resource_limits": {
            "cpu_count": 1.0,
            "memory_mb": 512,
            "timeout_seconds": 10
        }
    }"#;
    let plan_path = "plan.test.success.json";
    fs::write(plan_path, plan_json).unwrap();

    let output = Command::new("cargo")
        .args(&["run", "--", "execute", "--plan", plan_path, "--json"])
        .output()
        .expect("failed to execute process");

    fs::remove_file(plan_path).unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("\"status\": \"succeeded\""));
    assert!(stdout.contains("\"stdout\": \"hello\\n\""));
}

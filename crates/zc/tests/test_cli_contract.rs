use std::fs;
use std::process::Command;
use tempfile::tempdir;

#[test]
fn test_zc_execute_echo_success() {
    let tmp = tempdir().expect("Failed to create temp dir");
    let plan_path = tmp.path().join("plan.json");

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
    fs::write(&plan_path, plan_json).unwrap();

    // Use the absolute path for the plan file.
    let absolute_plan_path = fs::canonicalize(&plan_path).unwrap();

    // Use the current executable's path to find the `zc` binary or just run the relative command
    // from the project root. Since cargo is needed, let's execute `cargo run` from the project root.
    let output = Command::new("cargo")
        .args([
            "run",
            "-p",
            "zc",
            "--",
            "execute",
            "--plan",
            absolute_plan_path.to_str().unwrap(),
            "--json",
        ])
        .output()
        .expect("failed to execute process");

    if !output.status.success() {
        eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
    }

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("\"status\": \"succeeded\""));
    assert!(stdout.contains("\"stdout\": \"hello\\n\""));

    // Verify that the zakuro-artifacts directory is cleaned up
    let artifact_root = tmp.path().join("zakuro-artifacts");
    assert!(
        !artifact_root.exists(),
        "Artifacts directory should be cleaned up after execution"
    );
}

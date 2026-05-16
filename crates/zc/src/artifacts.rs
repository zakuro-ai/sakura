use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

pub fn prepare_artifact_dir(root_str: &str, job_id: &str) -> Result<PathBuf> {
    let root = Path::new(root_str);
    let artifact_dir = root.join(job_id);
    
    fs::create_dir_all(&artifact_dir)
        .with_context(|| format!("Failed to create artifact directory: {:?}", artifact_dir))?;
    
    let workspace_dir = artifact_dir.join("workspace");
    fs::create_dir_all(&workspace_dir)
        .with_context(|| format!("Failed to create workspace directory: {:?}", workspace_dir))?;

    Ok(artifact_dir)
}

pub fn write_artifact(artifact_dir: &Path, filename: &str, content: &str) -> Result<PathBuf> {
    let file_path = artifact_dir.join(filename);
    fs::write(&file_path, content)
        .with_context(|| format!("Failed to write artifact file: {:?}", file_path))?;
    Ok(file_path)
}

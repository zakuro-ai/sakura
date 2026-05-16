use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Execute an execution plan
    Execute {
        /// Path to the JSON execution plan
        #[arg(long)]
        plan: PathBuf,

        /// Output the result as JSON
        #[arg(long)]
        json: bool,
    },
}

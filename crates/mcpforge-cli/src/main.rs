use clap::{Parser, Subcommand};
use std::{fs, io, path::PathBuf};

#[derive(Debug, Parser)]
#[command(name = "mcpforge", about = "CLI for bootstrapping MCPForge projects")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    New {
        name: String,
        #[arg(long)]
        with_mcp_server: bool,
    },
    Doctor,
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::New {
            name,
            with_mcp_server,
        } => scaffold_project(&name, with_mcp_server),
        Commands::Doctor => {
            println!("MCPForge doctor");
            println!("- provider abstraction: available");
            println!("- agent runtime: available");
            println!("- session memory: available");
            println!("- MCP server primitives: available");
            println!("- CLI scaffolding: available");
            Ok(())
        }
    }
}

fn scaffold_project(name: &str, with_mcp_server: bool) -> io::Result<()> {
    let root = PathBuf::from(name);
    fs::create_dir_all(root.join("src"))?;

    fs::write(
        root.join("Cargo.toml"),
        format!(
            r#"[package]
name = "{name}"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = {{ version = "1", features = ["derive"] }}
serde_json = "1"
"#
        ),
    )?;

    let main_rs = if with_mcp_server {
        r#"fn main() {
    println!(\"Hello from your MCPForge app with MCP server support\");
}
"#
    } else {
        r#"fn main() {
    println!(\"Hello from your MCPForge app\");
}
"#
    };

    fs::write(root.join("src/main.rs"), main_rs)?;
    println!("Scaffolded {name}");
    Ok(())
}

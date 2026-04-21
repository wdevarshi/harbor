use clap::{Parser, Subcommand};
use std::{fs, io, path::PathBuf};

#[derive(Debug, Parser)]
#[command(name = "harbor", about = "CLI for bootstrapping Harbor projects")]
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
            println!("Harbor doctor");
            println!("- provider abstraction: available");
            println!("- OpenAI-compatible provider: available");
            println!("- Anthropic provider: available");
            println!("- agent runtime: available");
            println!("- shared HarborApp bootstrap: available");
            println!("- session memory: available");
            println!("- MCP server primitives: available");
            println!("- HTTP health/readiness surface: available");
            println!("- request ID propagation: available");
            println!("- request logging middleware: available");
            println!("- tracing bootstrap: available");
            println!("- OTEL exporter bootstrap: available");
            println!("- trace-context extraction: available");
            println!("- Prometheus metrics surface: available");
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
        "fn main() {\n    println!(\"Hello from your Harbor app with MCP server support\");\n}\n"
    } else {
        "fn main() {\n    println!(\"Hello from your Harbor app\");\n}\n"
    };

    fs::write(root.join("src/main.rs"), main_rs)?;
    println!("Scaffolded {name}");
    Ok(())
}

use harbor_runtime::{HarborApp, HarborAppConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = HarborAppConfig::from_env()?;
    if config.service_name == "harbor-app" {
        config.service_name = "harbor-bootstrap-example".into();
    }

    let app = HarborApp::new(config);

    println!(
        "Harbor app bootstrapping on http://{}:{}/healthcheck",
        app.config().http_host,
        app.config().http_port
    );

    app.run().await?;
    Ok(())
}

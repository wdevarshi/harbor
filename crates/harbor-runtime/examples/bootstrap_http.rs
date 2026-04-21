use harbor_runtime::{HarborApp, HarborAppConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = HarborApp::new(HarborAppConfig {
        service_name: "harbor-bootstrap-example".into(),
        service_version: "0.1.0".into(),
        environment: "dev".into(),
        http_host: "0.0.0.0".into(),
        http_port: 3000,
    });

    println!(
        "Harbor app bootstrapping on http://{}:{}/healthcheck",
        app.config().http_host,
        app.config().http_port
    );

    app.run().await?;
    Ok(())
}

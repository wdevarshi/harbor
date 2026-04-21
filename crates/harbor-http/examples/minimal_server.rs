use harbor_http::{HarborHttpConfig, HarborHttpServer};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = HarborHttpConfig {
        service_name: "harbor-example".into(),
        service_version: "0.1.0".into(),
        environment: "dev".into(),
        ..HarborHttpConfig::default()
    };

    println!(
        "Harbor HTTP server listening on http://{}:{}/healthcheck",
        config.host, config.port
    );

    HarborHttpServer::new(config).run().await?;
    Ok(())
}

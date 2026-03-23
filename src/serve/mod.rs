pub mod engine;
pub mod handlers;
pub mod types;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use axum::{routing::get, routing::post, Router};
use tokio::sync::mpsc;
use tower_http::cors::CorsLayer;

use crate::config::Config;
use engine::RealtimeEngine;
use handlers::AppState;

pub async fn start_server(
    configs: Vec<Config>,
    host: String,
    port: u16,
    tokenizer_path: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load tokenizer if provided
    let tokenizer = if let Some(path) = tokenizer_path {
        Some(Arc::new(
            tokenizers::Tokenizer::from_file(&path)
                .map_err(|e| format!("Failed to load tokenizer: {}", e))?,
        ))
    } else {
        None
    };

    let mut engines: HashMap<String, mpsc::Sender<types::EngineRequest>> = HashMap::new();
    let mut model_names: Vec<String> = Vec::new();

    for config in configs {
        let model_name = config.model.name.clone();

        // Create engine channel
        let (engine_tx, engine_rx) = mpsc::channel::<types::EngineRequest>(256);

        // Start the engine
        let engine = RealtimeEngine::new(config, engine_rx)?;
        tokio::spawn(engine.run());

        println!("  Loaded model: {}", model_name);
        model_names.push(model_name.clone());
        engines.insert(model_name, engine_tx);
    }

    // Build app state
    let state = Arc::new(AppState {
        engines,
        model_names: model_names.clone(),
        tokenizer,
    });

    // Build router
    let app = Router::new()
        .route("/health", get(handlers::health))
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", host, port);
    println!("Starting inference-lab server");
    println!("  Models: {}", model_names.join(", "));
    println!("  Listening on: http://{}", addr);
    println!();

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

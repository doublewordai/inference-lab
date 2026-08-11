pub mod directive;
pub mod engine;
pub mod fault;
pub mod handlers;
pub mod types;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use axum::{routing::get, routing::post, Router};
use hyper_util::rt::{TokioExecutor, TokioIo};
use tokio::sync::mpsc;
use tower::ServiceExt;
use tower_http::cors::CorsLayer;

use crate::config::{Config, ModelCosts};
use engine::RealtimeEngine;
use handlers::AppState;

pub async fn start_server(
    configs: Vec<Config>,
    host: String,
    port: u16,
    tokenizer_path: Option<PathBuf>,
    enable_directives: bool,
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
    let mut model_faults: HashMap<String, fault::FaultSpec> = HashMap::new();

    for config in configs {
        let model_name = config.model.name().to_string();

        // Validate static fault config at boot: a typo'd mode must fail the server, not
        // silently serve healthy streams under an e2e test that expects deaths.
        if let Some(fault_cfg) = &config.fault {
            let spec = fault::FaultSpec::from_config(fault_cfg)
                .map_err(|e| format!("invalid [fault] config for model {}: {}", model_name, e))?;
            println!(
                "  Static fault injection: {} (after_chunks={}) on {}",
                spec.mode.as_str(),
                spec.after_chunks,
                model_name
            );
            model_faults.insert(model_name.clone(), spec);
        }

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
        enable_directives,
        model_faults,
    });
    if enable_directives {
        println!("  Echo-directives: ENABLED (scripted-response test mode)");
    }

    // Build router
    let app = Router::new()
        .route("/health", get(handlers::health))
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/completions", post(handlers::completions))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", host, port);
    println!("Starting inference-lab server");
    println!("  Models: {}", model_names.join(", "));
    println!("  Listening on: http://{}", addr);
    println!();

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    // Manual accept loop instead of `axum::serve`: fault injection's `reset` mode needs
    // the raw TCP socket (SO_LINGER=0 abortive close), which axum never exposes. Each
    // request carries a `fault::FaultConnection` handle for its socket via extensions.
    // Non-fault traffic is served exactly as before — this is the same hyper auto
    // (HTTP/1 + HTTP/2) connection handling `axum::serve` uses internally.
    loop {
        let (socket, _remote_addr) = match listener.accept().await {
            Ok(accepted) => accepted,
            Err(e) => {
                // Same policy as `axum::serve`: per-connection errors (aborted handshakes
                // etc.) retry immediately; anything else (EMFILE/ENFILE) must not kill the
                // server, but backs off so a persistent error can't spin the loop.
                if !is_connection_error(&e) {
                    log::warn!("accept failed: {e}");
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
                continue;
            }
        };
        // A failed dup only costs the `reset` mode its RST (degrades to FIN, which the
        // fault path already tolerates) — never the connection itself.
        let conn = match fault::FaultConnection::new(&socket) {
            Ok(conn) => Some(Arc::new(conn)),
            Err(e) => {
                log::warn!("fault: could not dup connection fd ({e}); reset degrades to FIN");
                None
            }
        };
        let tower_service = app.clone();
        tokio::spawn(async move {
            let socket = TokioIo::new(socket);
            let hyper_service = hyper::service::service_fn(
                move |mut request: hyper::Request<hyper::body::Incoming>| {
                    if let Some(conn) = &conn {
                        request.extensions_mut().insert(conn.clone());
                    }
                    tower_service.clone().oneshot(request)
                },
            );
            if let Err(e) = hyper_util::server::conn::auto::Builder::new(TokioExecutor::new())
                .serve_connection_with_upgrades(socket, hyper_service)
                .await
            {
                // Fault modes abort response bodies on purpose; those deaths surface
                // here as connection errors and are expected.
                log::debug!("connection ended with error: {e}");
            }
        });
    }
}

/// Errors scoped to one failed connection attempt, not the listener — matches the set
/// `axum::serve` retries without logging or backoff.
fn is_connection_error(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::ConnectionRefused
            | std::io::ErrorKind::ConnectionAborted
            | std::io::ErrorKind::ConnectionReset
    )
}

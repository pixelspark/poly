use crate::config::Config;
use std::sync::Arc;
use tokio::sync::mpsc::{channel, Sender};

use poly_backend::backend::Backend;

pub struct Server {
	pub backend: Arc<Backend>,
	pub config: Config,
	ingest_sender: Sender<IngestItem>,
}

#[derive(Debug)]
pub struct IngestItem {
	pub memory_name: String,
	pub plaintext: String,
}

impl Server {
	pub fn new(backend: Arc<Backend>, config: Config) -> Self {
		// Queue for ingest
		let ingest_backend = backend.clone();
		let (tx, mut rx) = channel::<IngestItem>(32);
		tokio::spawn(async move {
			tracing::info!("starting ingest worker");
			while let Some(item) = rx.recv().await {
				tracing::trace!(?item, "ingest");
				match ingest_backend.memorize(&item.memory_name, &item.plaintext).await {
					Ok(_) => {}
					Err(e) => tracing::error!("error memorizing: {e}"),
				}
			}
			tracing::info!("ending ingest worker");
		});

		Server {
			backend,
			config,
			ingest_sender: tx,
		}
	}

	/// Enqueue an item for ingest
	pub async fn ingest(&self, item: IngestItem) {
		self.ingest_sender.send(item).await.unwrap()
	}
}

mod hora;

#[cfg(feature = "qdrant")]
mod qdrant;

use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::config::MemoryConfig;

#[derive(Debug, Error)]
pub enum MemoryError {
	#[error("mismatch in dimensionality")]
	DimensionalityMismatch,

	#[error("storage error: {0}")]
	Storage(String),
}

#[async_trait]
pub trait Memory: Send + Sync {
	/// Store the provided chunk in the memory
	async fn store(&self, text: &str, embedding: &[f32]) -> Result<(), MemoryError>;

	/// Retrieve relevant chunks from memory given an embedding. At most `top_n` chunks will be returned
	async fn get(&self, embedding: &[f32], top_n: usize) -> Result<Vec<String>, MemoryError>;
}

#[derive(Deserialize, Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStoreConfig {
	Hora {
		/// Path to the memory file (no path means not persisted)
		path: Option<PathBuf>,
	},

	#[cfg(feature = "qdrant")]
	Qdrant {
		/// URL to the Qdrant server
		#[serde(default = "default_qdrant_url")]
		url: String,

		/// Name of the collection
		collection: String,
	},
}

#[cfg(feature = "qdrant")]
fn default_qdrant_url() -> String {
	String::from("http://localhost:6333")
}

impl MemoryStoreConfig {
	pub fn from(&self, memory_config: &MemoryConfig) -> Result<Box<dyn Memory>, MemoryError> {
		match self {
			Self::Hora { path } => Ok(Box::new(hora::HoraMemory::new(path.clone(), memory_config.dimensions)?)),

			#[cfg(feature = "qdrant")]
			Self::Qdrant { url, collection } => Ok(Box::new(qdrant::QdrantMemory::new(url, collection, memory_config.dimensions)?)),
		}
	}
}

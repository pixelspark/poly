mod hora;

#[cfg(feature = "qdrant")]
mod qdrant;

use std::path::PathBuf;

use async_trait::async_trait;
use llm::TokenId;
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

	/// Clear the memory
	async fn clear(&self) -> Result<(), MemoryError>;
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

type TokenWithCharacters = (Vec<u8>, TokenId);

/// Apply successive separators to a chunk of text until it fits in a specific number of tokens. When there is no
/// separator anymore, just chunk.
pub fn hierarchically_chunk(tokens: Vec<TokenWithCharacters>, separators: &[TokenId], max_chunk_tokens: usize) -> Vec<Vec<TokenWithCharacters>> {
	tracing::trace!(n_tokens = tokens.len(), ?separators, max_chunk_tokens, "hierarchically chunk");
	// If the full chunk is small enough, no need to split anything
	if tokens.len() <= max_chunk_tokens {
		vec![tokens]
	} else {
		// We are too large to fit a single chunk; split until we fit
		if let Some(separator) = separators.first() {
			let next_separators = &separators[1..];
			let mut chunks = vec![];
			let mut current_chunk: Vec<TokenWithCharacters> = vec![];
			for split in tokens.split_inclusive(|x| x.1 == *separator) {
				if split.len() > max_chunk_tokens {
					// Can never make a chunk from this that is small enough
					assert!(current_chunk.len() <= max_chunk_tokens);
					chunks.push(std::mem::take(&mut current_chunk));
					chunks.append(&mut hierarchically_chunk(split.to_vec(), next_separators, max_chunk_tokens));
				} else if split.len() + current_chunk.len() < max_chunk_tokens {
					// Can append this split to the current chunk
					current_chunk.extend_from_slice(split);
					assert!(current_chunk.len() <= max_chunk_tokens);
				} else {
					// Make a new chunk
					chunks.push(std::mem::take(&mut current_chunk));
					assert_eq!(current_chunk.len(), 0);
					current_chunk.extend_from_slice(split);
					assert!(current_chunk.len() <= max_chunk_tokens);
				}
			}
			if !current_chunk.is_empty() {
				chunks.push(current_chunk);
			}
			chunks
		} else {
			// No more separators, just split by max size
			tokens.chunks(max_chunk_tokens).map(|x| x.to_vec()).collect()
		}
	}
}

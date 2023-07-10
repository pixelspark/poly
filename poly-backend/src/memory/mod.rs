pub mod hora;

use async_trait::async_trait;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemoryError {
	#[error("mismatch in dimensionality")]
	DimensionalityMismatch,
}

#[async_trait]
pub trait Memory: Send + Sync {
	/// Store the provided chunk in the memory
	async fn store(&self, text: &str, embedding: &[f32]) -> Result<(), MemoryError>;

	/// Retrieve relevant chunks from memory given an embedding. At most `top_n` chunks will be returned
	async fn get(&self, embedding: &[f32], top_n: usize) -> Result<Vec<String>, MemoryError>;
}

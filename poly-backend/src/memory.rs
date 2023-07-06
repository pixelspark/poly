use async_trait::async_trait;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemoryError {
	#[error("mismatch in dimensionality")]
	DimensionalityMismatch,
}

#[async_trait]
pub trait Memory: Send {
	async fn store(&mut self, text: &str, embedding: &[f32]) -> Result<(), MemoryError>;
	async fn get(&self, embedding: &[f32], top_n: usize) -> Result<Vec<String>, MemoryError>;
}

pub mod hora {
	use std::path::PathBuf;

	use crate::memory::{Memory, MemoryError};
	use async_trait::async_trait;
	use hora::core::ann_index::ANNIndex;
	use hora::core::ann_index::SerializableIndex;
	use hora::index::hnsw_idx::HNSWIndex;
	use hora::index::hnsw_params::HNSWParams;

	pub struct HoraMemory {
		path: PathBuf,
		index: HNSWIndex<f32, String>,
	}

	impl HoraMemory {
		pub fn new(path: &PathBuf, dims: usize) -> Result<HoraMemory, MemoryError> {
			let index = if path.exists() {
				HNSWIndex::<f32, String>::load(path.to_str().unwrap()).unwrap()
			} else {
				HNSWIndex::<f32, String>::new(dims, &HNSWParams::<f32>::default())
			};

			if index.dimension() != dims {
				return Err(MemoryError::DimensionalityMismatch);
			}

			Ok(HoraMemory { index, path: path.clone() })
		}
	}

	impl Drop for HoraMemory {
		fn drop(&mut self) {
			self.index.dump(self.path.to_str().unwrap()).unwrap();
		}
	}

	#[async_trait]
	impl Memory for HoraMemory {
		async fn store(&mut self, text: &str, embedding: &[f32]) -> Result<(), MemoryError> {
			assert_eq!(embedding.len(), self.index.dimension());
			// TODO: error handling
			self.index.add(embedding, text.to_string()).unwrap();
			self.index.build(hora::core::metrics::Metric::Euclidean).unwrap();
			self.index.dump(self.path.to_str().unwrap()).unwrap();
			Ok(())
		}

		async fn get(&self, embedding: &[f32], top_n: usize) -> Result<Vec<String>, MemoryError> {
			assert_eq!(embedding.len(), self.index.dimension());
			Ok(self.index.search(embedding, top_n))
		}
	}

	#[cfg(test)]
	mod test {
		use super::HoraMemory;
		use crate::memory::Memory;

		#[tokio::test]
		pub async fn test_store() {
			let mut hm = HoraMemory::new(3);
			hm.store("foo", &[1.0, 2.0, 3.0]).await.unwrap();
			hm.store("bar", &[-1.0, 2.0, 3.0]).await.unwrap();
			hm.store("baz", &[1.0, -2.0, 3.0]).await.unwrap();
			hm.store("boo", &[1.0, -2.0, -3.0]).await.unwrap();
			assert_eq!(hm.get(&[0.0, -1.0, 0.0], 2).await.unwrap(), vec!["baz", "boo"]);
		}
	}
}

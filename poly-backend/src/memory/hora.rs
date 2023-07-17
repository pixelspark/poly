use std::path::PathBuf;

use crate::memory::{Memory, MemoryError};
use async_trait::async_trait;
use hora::core::ann_index::ANNIndex;
use hora::core::ann_index::SerializableIndex;
use hora::index::hnsw_idx::HNSWIndex;
use hora::index::hnsw_params::HNSWParams;
use tokio::sync::Mutex;

pub struct HoraMemory {
	path: Option<PathBuf>,
	index: Mutex<HNSWIndex<f32, String>>,
}

impl HoraMemory {
	pub fn new(path: Option<PathBuf>, dims: usize) -> Result<HoraMemory, MemoryError> {
		let index = if let Some(ref path) = path {
			if path.exists() {
				HNSWIndex::<f32, String>::load(path.to_str().unwrap()).unwrap()
			} else {
				HNSWIndex::<f32, String>::new(dims, &HNSWParams::<f32>::default())
			}
		} else {
			tracing::warn!("creating a memory store that is non-persistent");
			HNSWIndex::<f32, String>::new(dims, &HNSWParams::<f32>::default())
		};

		if index.dimension() != dims {
			return Err(MemoryError::DimensionalityMismatch);
		}

		Ok(HoraMemory {
			index: Mutex::new(index),
			path,
		})
	}
}

impl Drop for HoraMemory {
	fn drop(&mut self) {
		if let Some(ref path) = self.path {
			self.index.blocking_lock().dump(path.to_str().unwrap()).unwrap();
		}
	}
}

#[async_trait]
impl Memory for HoraMemory {
	async fn store(&self, text: &str, embedding: &[f32]) -> Result<(), MemoryError> {
		let mut index = self.index.lock().await;
		assert_eq!(embedding.len(), index.dimension());
		// TODO: error handling
		index.add(embedding, text.to_string()).unwrap();
		index.build(hora::core::metrics::Metric::Euclidean).unwrap();
		if let Some(ref path) = self.path {
			index.dump(path.to_str().unwrap()).unwrap();
		}
		Ok(())
	}

	async fn get(&self, embedding: &[f32], top_n: usize) -> Result<Vec<String>, MemoryError> {
		let index = self.index.lock().await;
		assert_eq!(embedding.len(), index.dimension());
		Ok(index.search(embedding, top_n))
	}

	async fn clear(&self) -> Result<(), MemoryError> {
		let mut index = self.index.lock().await;
		index.clear();
		if let Some(ref path) = self.path {
			index.dump(path.to_str().unwrap()).unwrap();
		}
		Ok(())
	}
}

#[cfg(test)]
mod test {
	use super::HoraMemory;
	use crate::memory::Memory;

	#[tokio::test]
	pub async fn test_store() {
		let hm = HoraMemory::new(None, 3).unwrap();
		hm.store("foo", &[1.0, 2.0, 3.0]).await.unwrap();
		hm.store("bar", &[-1.0, 2.0, 3.0]).await.unwrap();
		hm.store("baz", &[1.0, -2.0, 3.0]).await.unwrap();
		hm.store("boo", &[1.0, -2.0, -3.0]).await.unwrap();
		assert_eq!(hm.get(&[0.0, -1.0, 0.0], 2).await.unwrap(), vec!["baz", "boo"]);
	}
}

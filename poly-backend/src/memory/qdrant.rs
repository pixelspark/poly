use async_trait::async_trait;
use qdrant_client::{prelude::*, qdrant::PointsSelector};
use serde_json::json;

use super::{Memory, MemoryError};

pub struct QdrantMemory {
	client: QdrantClient,
	collection_name: String,
	dimensions: usize,
}

impl QdrantMemory {
	pub fn new(url: &str, collection_name: &str, dimensions: usize) -> Result<QdrantMemory, MemoryError> {
		let config = QdrantClientConfig::from_url(url);
		let client = QdrantClient::new(Some(config)).map_err(|x| MemoryError::Storage(x.to_string()))?;
		Ok(QdrantMemory {
			client,
			collection_name: collection_name.to_string(),
			dimensions,
		})
	}
}

const ITEM_NAMESPACE: uuid::Uuid = uuid::uuid!("067FB304-F9B1-4E74-8ACA-28051B8492AB");

#[async_trait]
impl Memory for QdrantMemory {
	async fn store(&self, text: &str, embedding: &[f32]) -> Result<(), MemoryError> {
		assert_eq!(
			embedding.len(),
			self.dimensions,
			"embedding to store must have same dimensionality as configured for the memory"
		);
		let payload: Payload = json!({ "text": text }).try_into().unwrap();
		let id = uuid::Uuid::new_v5(&ITEM_NAMESPACE, text.as_bytes());
		let points = vec![PointStruct::new(id.to_string(), embedding.to_vec(), payload)];
		self.client
			.upsert_points_blocking(&self.collection_name, None, points, None)
			.await
			.map_err(|x| MemoryError::Storage(x.to_string()))?;
		Ok(())
	}

	async fn get(&self, embedding: &[f32], top_n: usize) -> Result<Vec<String>, MemoryError> {
		assert_eq!(
			embedding.len(),
			self.dimensions,
			"embedding to search must have same dimensionality as configured for the memory"
		);
		let search_result = self
			.client
			.search_points(&SearchPoints {
				collection_name: self.collection_name.to_string(),
				vector: embedding.to_vec(),
				filter: None,
				limit: top_n as u64,
				with_payload: Some(true.into()),
				..Default::default()
			})
			.await
			.map_err(|x| MemoryError::Storage(x.to_string()))?;

		Ok(search_result.result.into_iter().map(|r| r.payload["text"].to_string()).collect())
	}

	async fn clear(&self) -> Result<(), MemoryError> {
		self.client
			.delete_points(self.collection_name.to_string(), None, &PointsSelector::default(), None)
			.await
			.map_err(|x| MemoryError::Storage(x.to_string()))?;
		Ok(())
	}
}

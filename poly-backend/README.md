# Poly-backend

## Usage

### With Qdrant

#### Set up Qdrant

Run Qdrant with the gRPC service enabled:

```sh
docker pull qdrant/qdrant
docker run -p 6333:6333 -e QDRANT__SERVICE__GRPC_PORT="6334" qdrant/qdrant
```

Or, when building Qdrant from source, create a config file:

```yaml
service:
  grpc_port: 6334
```

Then run:

```sh
cargo run --release -- --config-path=./data/qdrant.yaml --disable-telemetry
```

#### Prepare a collection

Create a collection:

```sh
curl -XPUT http://localhost:6333/collections/test -vvv -d '{"vectors":{"size":3200,"distance":"Cosine"}}' -H "Content-type: application/json"
```

#### Configure Poly to use Qdrant

Note the port number refers to the gRPC port - you will see errors relating to invalid HTTP versions when using the HTTP/REST port.

```toml
[memories.qdrant_test]
embedding_model = "orcamini3b"
dimensions = 3200
store = { qdrant = { url = "http://localhost:6334", collection = "test" } }
chunk_separators = ["."]
chunk_max_tokens = 255
```

### Model caching

If a `url` is specified in the model configuration, the model file will be automatically downloaded from the specified URL
if it cannot be found locally (either at the specified `model_path` or in the platform-specific cache directory).

Notes:

- Models are stored in the cache directory by their model key. If the model key changes, the model may be redownloaded.
- Currently, resuming incomplete downloads is not supported, but incomplete downloads may be left in the cache directory.

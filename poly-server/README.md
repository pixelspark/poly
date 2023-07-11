## LLMD

Server for serving LLM models over HTTP using GGML as backend.

## Usage

Adapt `config.example.toml` to your needs and save as `config.toml`, then:

```sh
cargo run --release
```

To enable logging:

```sh
RUST_LOG=llmd=debug cargo run --release
```

```ps
$env:RUST_LOG="llmd=debug"
cargo run --release
```

### API

To generate completions:

```sh
curl -XPOST --url http://localhost:3000/v1/task/pythia/completion --header 'Content-type: application/json' --data '{"prompt": "Hello "}' -vvv
```

To stream completions as they are generated:

```sh
curl --url "http://localhost:3000/v1/task/pythia/live?prompt=foo&max_tokens=10" -vvv
```

To generate embeddings:

```sh
curl -XPOST --url http://localhost:3000/v1/task/pythia/embedding --header 'Content-type: application/json' --data '{"prompt": "Hello "}' -vvv
```

#### Memories

```sh
curl -XPUT "http://localhost:3000/v1/memory/test?api_key=foo" -vvvv -d "Hello, world" -H "Content-type: text/plain"
```

See [openapi.yaml](./openapi.yaml) for more information (incomplete as of yet).

### With Qdrant

Build with feature `qdrant` enabled, then configure:

```toml
[memories.qtest]
store = { qdrant = { url = "http://localhost:6334", collection = "test" } }
dimensions = 3200
embedding_model = "orcamini3b"
```

See [poly-backend](../poly-backend/README.md) for further information.

### With CUDA

From the root of the repository:

```bash
sudo docker build -t llmd -f cublas.Dockerfile .
sudo docker run -it --rm -v $(pwd)/data:/llmd/data -v $(pwd)/config.toml:/llmd/config.toml --gpus all -e RUST_LOG=debug -p 3000:3000 llmd
```

### WebSocket chat API

To chat, connect through WebSocket to the following endpoint:

`ws://localhost:3000/v1/task/pythia/chat?api_key=<key>`

Send messages as text frames, and receive individual token messages. When a message is finished, the server will send an
empty text frame.

### Securing the API

Unless `public` is set to `true` in config, access is only granted when a valid API key is provided. Depending on configuration
this can be a pre-shared static API key and/or a JWT token.

API keys need to be supplied either using an `Authorization: Bearer <key>` header, or using a `?api_key=<key>` query parameter.
When both are supplied the header key takes precedence.

#### Static pre-shared keys

To use static pre-shared API keys, add the allowed API keys to the config file (`allowed_keys`).

#### JWT tokens

To use JWT tokens, first configure a shared secret key in config. Currently only a symmetric key is supported:

```toml
jwt_private_key = { symmetric = "..." }
```

Generated tokens should use the `HS256` algorithm and have an expiry time set (`exp`). If an `nbf` (not valid before) time
is present, it will be validated.

To generate a token for testing, use `cargo run --bin token` (this token by default expires in an hour).

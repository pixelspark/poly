## LLMD

Server for serving LLM models over HTTP using GGML as backend.

Author: Tommy van der Vorst (vandervorst@dialogic.nl)

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

### WebSocket chat API

To chat, connect through WebSocket to the following endpoint:

`ws://localhost:3000/v1/task/pythia/chat?api_key=<key>`

Send messages as text frames, and receive individual token messages. When a message is finished, the server will send an
empty text frame.

### Securing the API

To limit access to specific users, add a set of allowed API keys to the config file (`allowed_keys`). Then supply the key
on each request, either using an `Authorization: Bearer <key>` header, or using a `?api_key=<key>` query parameter. When
both are supplied the header key takes precedence.

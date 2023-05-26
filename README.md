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
curl -XPOST --url http://10.10.1.160:3000/pythia/completion --header 'Content-type: application/json' --data '{"prompt": "Hello "}' -vvv
```

To stream completions as they are generated:

```sh
curl -XPOST --url http://10.10.1.160:3000/pythia/live --header 'Content-type: application/json' --data '{"prompt": "Hello "}' -vvv
```

# LLM-ui

Place a model in the data folder and edit [config.toml](./data/config.toml). (Note, all model paths are relative to the
data directory, except those that start with a "/").

```sh
cargo run --release --bin llm-ui --features=metal
```

## Building installers

```sh
cargo add --dev cargo-bundle
cargo bundle --release --features=metal
```

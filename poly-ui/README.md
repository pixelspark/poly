# Poly-ui

Place a model in the data folder and edit [config.toml](./data/config.toml). (Note, all model paths are relative to the
data directory, except those that start with a "/").

```sh
cargo run --release --bin poly-ui --features=metal
```

## Building installers

```sh
cargo add --dev cargo-bundle
cargo bundle --release --features=metal
```

## Usage

To use different models than the one included, you can place a custom `config.toml` in one of the following places:

- macOS: `~/Library/Application Support/nl.Dialogic.Poly/_config.toml`
- Linux: `~/.config/poly/config.toml`
- Windows: `%USERPROFILE%\AppData\Roaming\Dialogic\Poly\config.toml`

When a file is found, it will be used over the included configuration file.

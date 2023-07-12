# HOWTOs

## Using custom GPT-2 embeddings

GPT-2 models can be perfect for embedding tasks. It is quite easy to find language specific ones (e.g.
[gpt2-small-dutch-embeddings](https://huggingface.co/GroNLP/gpt2-small-dutch-embeddings)). To use these with Poly, follow
the following steps:

1. Clone the GPT-2 model:

```sh
git clone https://huggingface.co/GroNLP/gpt2-small-dutch-embeddings
git lfs install
git lfs pull
```

2. Clone GGML:

```sh
git clone https://github.com/ggerganov/ggml.git
git lfs install
git lfs pull
```

3. Install requirements:

```sh
cd ggml
pip install -r requirements.txt
```

4. Check if the 'end of text' token in the GPT-2 model is set to `<|endoftext|>`. To do this, first open the model's
   `tokenizer_config.json` to find the `eos_token` (in the example linked above, the `eos_token` is set to `</s>`). If the
   `eos_token` is not set to `<|endoftext|>`, open the model's `vocab.json` and replace `</s>` (or whatever the `eos_token`
   key was set to) with `<|endoftext|>` (the token should be at the beginning of the vocab file).

5. Convert GPT-2 model to GGML format:

```sh
python3  ./examples/gpt-2/convert-cerebras-to-ggml.py ~/gpt2-small-dutch/
```

This results in a `ggml-model-f16.bin` file in the model directory that can be used with Poly.

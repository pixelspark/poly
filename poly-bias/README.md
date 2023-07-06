# poly-bias

LLM response biasing utilities.

This crate can be used to bias the output of an LLM, using e.g. a JSON schema. Before each token is generated, a biaser
can be used to calculate biases for the next expected tokens based on the grammar of the desired output. This influences
the sampling process so that only tokens that are valid according to the grammar are sampled. As the generated tokens also
become part of the LLM's context, the biaser nudges the output.

The process is as follows (using the `llm` crate):

- Load a model
- Feed initial prompt (use `feed_prompt`)
- Instantiate a `Biaser` (i.e. `JsonBiaser` using a JSON schema)
- For as many tokens as you need to generate (and/or until the biaser indicates there are no more valid next tokens):
  - Call `next_valid_tokens` on the biaser to obtain a set of token biases
  - If the set of token biases contains just one (positively biased) token
    - Append it to the result (feed as prompt)
    - Else infer the next token, supplying the token biases for use by the sampler
  - Call `advance` on the bias with the new token

See [./tests/biaser.rs](./tests/biaser.rs) for a usage example.

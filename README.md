# Poly

Versatile LLM back-end.

- [poly-server](./poly-server): Serve LLMs through HTTP and WebSocket APIs (provides `llmd`)
- [poly-backend](./poly-backend): Back-end implementation of LLM tasks
- [poly-bias](./poly-bias): Crate for biasing LLM output to e.g. JSON following a schema
- [poly-ui](./poly-ui): Simple desktop UI for local LLMs

```mermaid
flowchart TD

subgraph Poly
	PW[poly web-client]
	PS[poly-server]
	PB[poly-backend]
	PU[poly-ui]
	PBs[poly-bias]

	PW-->|HTTP,WS,SSE|PS
	PB-->PBs

	PS-->PB
	PU-->PB
end

LLM[rustformers/llm]
PBs<-.->LLM

T[tokenizers]
GGML
Metal
PB-->LLM
LLM<-.->T
LLM-->GGML
GGML-->Metal
GGML-->CUDA

```

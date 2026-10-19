# Streaming API

The API serves one resident model, named `eva`. It supports deterministic text generation through `/v1/chat/completions` and `/v1/completions` using an OpenAI-style JSON and SSE format.

## Stream a response

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"eva","messages":[{"role":"user","content":"Write a Python square function."}],"temperature":0,"max_tokens":128,"stream":true}'
```

The chat endpoint accepts one text user message per request. For `/v1/completions`, replace `messages` with a string `prompt`; both endpoints use the model's single-user chat template. `GET /v1/models` lists the model; `GET /health` reports loading, readiness and queue state.

Text arrives in `choices[0].delta.content` for chat or `choices[0].text` for completions. The final event includes `finish_reason`, `usage` and `metrics`, followed by `data: [DONE]`.

| Parameter | Meaning |
| --- | --- |
| `max_tokens` | Output-token limit; `max_completion_tokens` is also accepted |
| `temperature` | `0`, for deterministic generation |
| `stream` | `true` for SSE; `false` returns one completed JSON response |
| `mode` | `greedy` or `lookup` |
| `stop` | Up to four literal stop strings |

## Read the timings

| Field | Scope |
| --- | --- |
| `metrics.engine.first_token_s` | Worker request start through its first generated token, including reset and prefill |
| `metrics.engine.first_text_s` | Worker request start through the first nonempty text delta |
| `metrics.engine.decode_tokens` | Generated token count after the first token |
| `metrics.engine.decode_s` | Decode interval, including streaming callbacks and retirement |
| `metrics.engine.decode_tok_s` | `decode_tokens / decode_s`; null when there is no valid decode interval |
| `metrics.engine.e2e_s` | Worker request duration |
| `metrics.server.queue_s` | Time waiting for the worker |
| `metrics.server.first_text_delta_s` | HTTP acceptance through the first nonempty text delta |
| `metrics.server.e2e_s` | HTTP acceptance through worker completion |

The terminal client separately reports first-text and total time as observed across the connection. Prompt-driven expert placement can run after the first token and before the decode interval; its duration is reported as `prompt_placement_s`. Tokenization, queueing, placement, transport and the number of accepted proposals affect different parts of these timings; decode speed alone is not request latency.

## Limits and cancellation

Context and output limits are checked against the actual tokenizer. Oversized prompts are rejected rather than silently truncated. Each request is independent; applications can include the desired context in the prompt.

The worker processes one request at a time. A full queue returns HTTP 429; loading or unavailable workers return 503. Validation errors return 400. If an error occurs after streaming starts, it is sent as an SSE error before `[DONE]`.

Closing a connection requests cancellation. The next request waits until the current worker has retired its outstanding GPU and transfer work.

Set `EVA_API_KEY` on the server to require `Authorization: Bearer <key>`. The client reads the same environment variable when sending requests.

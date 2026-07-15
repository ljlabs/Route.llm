# Public API compatibility contract

**Last reviewed:** 2026-07-15
**Scope:** inbound endpoints implemented by this local router. This is a compatibility contract, not a claim of complete OpenAI or Anthropic API coverage.

## Upstream specifications

- [OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
- [OpenAI Responses](https://developers.openai.com/api/docs/guides/migrate-to-responses)
- [Anthropic Messages](https://docs.claude.com/en/api/messages)
- [Anthropic token counting](https://docs.claude.com/en/api/messages-count-tokens)

The summaries below are rephrased from those specifications for compliance with licensing restrictions. The implementation (`api/proxy.py`, `models/request.py`) and alignment suite remain the executable source of truth. Update this document whenever that public contract changes.

## Status labels

- **Supported** — exposed and covered by the router's compatibility tests.
- **Conditional** — accepted or translated by the router, but the configured upstream model/provider determines the final capability.
- **Not supported** — intentionally outside this router's public surface; do not rely on it being added implicitly.

## OpenAI compatibility

### Supported public surface

| Endpoint or feature | Status | Router behavior |
| --- | --- | --- |
| `POST /v1/chat/completions` and `POST /chat/completions` | Supported | Chat Completions requests and responses, including SSE when `stream: true`. |
| Chat messages | Supported | `system`, `developer`, `user`, `assistant`, and `tool` roles; text and supported content arrays. |
| Tools / legacy functions | Supported | Function schemas, tool calls, and tool-result turns are translated between protocols. |
| Multimodal inline data | Conditional | Base64 image data URIs plus base64 file/PDF inputs are translated when the selected upstream supports them. |
| `GET /v1/models`, `GET /models`, `GET /v1/models/{id}` | Supported | Lists router mappings and the active provider model, not the vendor's complete catalog. |
| `POST /v1/embeddings` | Conditional | Exposed by the router; support depends on the selected embedding provider. |

Public `Authorization: Bearer ...` headers are accepted but ignored. This local service does not authenticate callers, perform OpenAI organization/project handling, or validate caller API keys.

### Explicitly not supported

| OpenAI API or behavior | Decision |
| --- | --- |
| `POST /v1/responses` | Not exposed. The router's mock understands selected Responses-shaped content for translation testing only; it is not a public Responses API. |
| Legacy `POST /v1/completions` | Not exposed. |
| Assistants, Threads, Runs, vector stores, Batch, Fine-tuning, Files, Realtime, Audio, Images, Moderations, and administration/billing APIs | Not proxied or emulated. |
| Stored-completion retrieval/deletion and vendor resource IDs | Not implemented; router traffic is not an OpenAI resource store. |
| OpenAI credential, organization, project, rate-limit, usage, or billing semantics | Not implemented at the public boundary. |
| Guaranteed support for every Chat Completions option | Not promised. Unknown fields may be ignored, and provider-specific options may be dropped during cross-protocol translation. |

## Anthropic compatibility

### Supported public surface

| Endpoint or feature | Status | Router behavior |
| --- | --- | --- |
| `POST /v1/messages` | Supported | Messages requests and Anthropic-shaped non-streaming and SSE responses. |
| System prompts | Supported | Top-level `system` strings/blocks and Claude Code inline `messages[].role = "system"`; inline segments are normalized ahead of conversation messages. |
| Conversations | Supported | After system normalization, conversations begin with `user` and contain `user`/`assistant` turns. |
| Tools and tool results | Supported | Anthropic tool definitions, `tool_use`, and `tool_result` blocks are translated to/from OpenAI-style upstreams. |
| Multimodal inline data | Conditional | Base64 image blocks and base64 documents (PDF or generic file types) are translated when supported by the selected upstream. |
| `POST /v1/messages/count_tokens` | Supported with local semantics | Returns a deterministic local text-token estimate; it is not Anthropic's authoritative tokenizer. |
| `GET /v1/models` / `GET /models` with `anthropic-version` | Supported | Returns Anthropic-shaped model-list objects for router mappings and the active provider model. |

`x-api-key` and `anthropic-version` headers may be sent by Anthropic SDKs. The API-key value is ignored; `anthropic-version` is optional and defaults to `2023-06-01` when absent.

### Explicitly not supported

| Anthropic API or behavior | Decision |
| --- | --- |
| Message Batches, Files, Skills, Administration, Organization, usage/cost, and other vendor-management APIs | Not exposed or emulated. |
| Hosted file references and vendor file IDs | Not supported. Send needed file content inline as base64 in a request. |
| Anthropic API-key authentication, key lifecycle, workspace access, or billing enforcement | Not implemented; this is a local no-auth proxy. |
| `anthropic-beta` feature enablement and complete beta-feature parity | Not supported as a compatibility guarantee. |
| Provider-native server tools (for example, web search, code execution, or computer use) | Not emulated. Tool forwarding only supports caller-defined tool schemas and results. |
| Guaranteed prompt caching, citations, thinking, or every provider-specific content block across translation | Not guaranteed. These features are provider-dependent and may be omitted on a cross-protocol route. |

## Cross-protocol limits

The router translates the shared message, stream, tool, and inline-media subset; it cannot manufacture capabilities absent from the active upstream. Set model mappings deliberately and treat a response from a different provider as compatibility-shaped rather than vendor-identical. Add an alignment test before moving a capability from **Conditional** to **Supported**.

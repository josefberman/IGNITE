# Threat model (MVP)

## Assets

- Search jobs and candidate molecules
- Model weights and seed library
- Optional API keys (OpenAI, Lens, EPO OPS)

## Threats and mitigations

| Threat | Mitigation |
|--------|------------|
| Resource exhaustion via huge beams | Hard caps `IGNITE_MAX_BEAM_*` |
| Path traversal via uploads | No arbitrary file upload endpoints |
| Secret leakage in exports | Exports contain molecules/scores only |
| Over-trust of “safe/novel” | Explicit disclaimers; incomplete provider status |
| SSRF via user URLs | No user-supplied URL fetch |

## Operational limits

- In-process threaded jobs (Redis optional later)
- Network novelty providers time out and degrade gracefully

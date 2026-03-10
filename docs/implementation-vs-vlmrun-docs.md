# Our Implementation vs VLM Run Docs: Comparison

## How VLM Run docs say to do it

The docs show **two patterns** depending on use case:

### Pattern 1 — Generation (image/video creation)

Define a simple Pydantic model with `url: str`, send its schema via `response_format`, parse with `model_validate_json()`, download from the URL directly.

```python
# From docs: agents/capabilities/image/generation
class ImageGenerationResponse(BaseModel):
    url: str = Field(..., description="The URL of the generated image")

response = client.agent.completions.create(
    model="vlmrun-orion-1:auto",
    messages=[...],
    response_format={"type": "json_schema", "schema": ImageGenerationResponse.model_json_schema()}
)
result = ImageGenerationResponse.model_validate_json(response.choices[0].message.content)
# result.url → direct download URL like "https://mintcdn.com/..."
```

Same pattern for video generation and video tools (trimming, highlight extraction) — all return `url: str` fields.

### Pattern 2 — Artifact refs (edit/tool outputs)

The artifacts page shows responses contain refs like `img_a1b2c3`, retrieved via `client.artifacts.get(session_id=..., object_id=...)`. The SDK provides typed `Ref` classes (`ImageRef`, `VideoRef`, etc.) with an `.id` field for this.

## How our code does it

### Generate mode (`_execute_generate`, line ~896)

- Sends **no** `response_format` — gets free text back
- Regex-scans the prose for refs like `vid_abc123` (`re.findall`)
- Downloads via `client.artifacts.get(session_id, object_id)`
- This works because the agent's natural language response happens to contain artifact refs, but it's fragile — depends on the agent's wording

### Edit mode (`_execute_edit`, line ~1104)

- Builds a JSON schema using `_build_artifact_schema()` with `ImageRef`/`VideoRef` types
- Sends it via `response_format`
- But then parses with `json.loads()` + manual dict walking (`_parse_artifact_ids`) instead of `model_validate_json()`
- Falls back to regex if parsing fails
- Downloads via `client.artifacts.get(session_id, object_id)` — correct

### Analyze/Annotate modes

Not artifact-related, working fine as-is.

## Key gaps

| Aspect | VLM Run Docs | Our Generate | Our Edit |
|--------|-------------|-------------|---------|
| `response_format` | Always sent with schema | Not sent | Sent (correct) |
| Schema type | `url: str` for generation, `Ref` types for artifacts | N/A | `Ref` types (correct) |
| Parsing | `model_validate_json()` | Regex on free text | `json.loads()` + dict walk + regex fallback |
| Model class retained | Yes (needed for `model_validate_json`) | N/A | No — `_build_artifact_schema` discards it, returns only schema dict |
| Download method | Direct URL for generation; `client.artifacts.get()` for refs | `client.artifacts.get()` | `client.artifacts.get()` (correct) |

## What's unclear

The docs show generation returning `url: str` (direct download URLs), but our generate mode successfully finds artifact refs (`vid_abc123`) in the free text and downloads them via `client.artifacts.get()`. Both patterns seem to work — the question is whether we should:

1. **Match the docs exactly**: Use `url: str` schema for generate mode, download directly from URL
2. **Use Ref types everywhere**: Send `ImageRef`/`VideoRef` schema for generate mode too, parse with `model_validate_json()`, keep using `client.artifacts.get()`

Either way, both modes should use `model_validate_json()` instead of manual JSON parsing + regex fallback.

## Sources

- [Artifacts](https://docs.vlm.run/agents/artifacts)
- [Structured Responses](https://docs.vlm.run/agents/structured-responses)
- [Image Generation](https://docs.vlm.run/agents/capabilities/image/generation)
- [Video Generation](https://docs.vlm.run/agents/capabilities/video/generation)
- [Video Tools](https://docs.vlm.run/agents/capabilities/video/tools)
- [Image Tools](https://docs.vlm.run/agents/capabilities/image/tools)
- [Agent SDK Reference](https://docs.vlm.run/sdk-reference/components/agent)

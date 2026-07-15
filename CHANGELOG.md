# Changelog

All notable changes to the VLM Run FiftyOne plugin are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0]

### Changed

- **Chat Completions now defaults to the Orion 2 model family**
  (`vlmrun-orion-2:auto`). Orion 2 agents are code-execution agents; the
  OpenAI-compatible chat-completions response contract is unchanged, so this is
  a drop-in model upgrade. This is a **breaking change** for anyone relying on
  the previous `vlmrun-orion-1:auto` default: existing operator runs that did
  not explicitly pin a model will now use Orion 2, which may differ in latency,
  cost, and behavior.
- Raised the minimum `vlmrun` SDK requirement to `>=0.6.5` (the version this
  release was validated against), which provides `client.agent.completions`,
  `client.artifacts`, and toolset support required by the Chat Completions
  operator.

### Added

- Expanded the Chat Completions model dropdown to match the chat-completions
  API enum: Orion 2 tier aliases (`lite|fast|auto|pro`), named Orion 2
  backends (Qwen, Gemma, Kimi, GPT, Opus, Muse Spark, Grok, Gemini Flash),
  and Orion 1 (`lite|fast|auto|pro`) for backward compatibility.
- `VLMRUN_DEFAULT_MODEL` environment variable to override the default model
  (e.g. pin an Orion 1 variant or a pinned backend variant), providing an
  opt-out from the Orion 2 default without editing code. Leading/trailing
  whitespace is stripped.
- Enabled **delegated (background) execution** for the Chat Completions
  operator (`allow_delegated_execution=True`). Long-running Orion operations —
  video edit/generation and document redaction — can exceed the synchronous
  execution limit; they can now be run as delegated jobs. The operator already
  handled `ctx.delegated` internally.

## [1.0.0]

- Initial release with object/person/layout detection, invoice parsing, image
  captioning, video transcription, and Orion chat completions operators.

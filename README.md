# Helix Model Runner

Disposable public compute adapter for Helix Model experiments.

This repository is **not** an authority for Helix, Helix Model, research claims, model identity, or promotion. Canonical experiment definitions and accepted evidence live in the private Helix Model repository. This repository exists only to obtain free, replaceable GitHub-hosted compute for experiments whose inputs are public.

## Rules

- No private Helix data, credentials, hidden benchmark answers, or judge state.
- No user-workstation model downloads.
- Model artifacts exist only on ephemeral hosted runners unless separately promoted.
- Every run pins public source identity and verifies artifact digests before use.
- Outputs are measurements/receipts, never self-authorizing claims.
- Workflows should remain reproducible and disposable; another free compute provider may replace this repository without changing Helix semantics.

The initial experiment mirrors Helix Model `013-real-substrate-structure-probe` and tests the exact pinned SmolLM2-360M base weights.

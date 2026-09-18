# Helix Model Runner

Disposable public compute adapter for Helix Model experiments.

This repository is **not** an authority for Helix, Helix Model, research claims, model identity, or promotion. Canonical experiment definitions and accepted evidence live in the private Helix Model repository. This repository exists only to obtain free, replaceable hosted compute for experiments whose inputs are public.

## Rules

- No private Helix data, credentials, hidden benchmark answers, or judge state.
- No user-workstation model downloads.
- Model bytes must not enter repository history or uploaded result artifacts.
- Model residency is transport state: bytes may exist in provider storage, ephemeral hosted runners, or disposable remote caches keyed by immutable actor identity.
- Remote cache or provider mounts are never authority. Every run pins public source identity and verifies the model artifact digest before use.
- Reuse retained evidence instead of rerunning an unchanged comparator merely to recreate a result.
- Outputs are compact measurements and receipts, never self-authorizing claims.
- Workflows should remain reproducible and disposable; another suitable compute provider may replace this repository without changing Helix semantics.

## Model transport

Prefer the cheapest transport that preserves exact actor identity:

```text
provider-native read-only model mount / stream
→ remote content-addressed cache
→ cold pinned remote fetch
→ never a workstation download unless the user explicitly opts in
```

Caches are acceleration only. Eviction must not affect reproducibility because the pinned upstream revision plus verified artifact digest remains sufficient to reacquire the actor remotely.

The initial experiment mirrors Helix Model `013-real-substrate-structure-probe` and tests the exact pinned SmolLM2-360M base weights.

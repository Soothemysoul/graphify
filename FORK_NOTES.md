# Isolated fork — agents-brain memory system

This fork of `safishamsi/graphify` is **isolated** for the agents-brain
project (see `~/repos/agents-brain-team/system/graphify-patches/`). It
is **not** intended to be synced back to upstream.

## Rules

- Do **NOT** add an `upstream` remote.
- Do **NOT** open PRs to `safishamsi/graphify`.
- The only remote allowed is `origin = Soothemysoul/graphify`.
- A `pre-push` git hook rejects pushes to any other remote (belt-and-
  braces in case someone adds `upstream` by mistake).

## Patches

All modifications live on branch `matching-aliases-enru`:

- `graphify/serve.py` — EN↔RU transliteration, aliases.json lookup with
  mtime-cache, `_find_node`/`_suggest_labels` improvements, match-path
  annotation in `query_graph` output.
- `tests/test_matching.py` — 29-test pytest suite for the above.

Source of truth for patches:
`~/repos/agents-brain-team/system/graphify-patches/`

To sync with upstream (if ever needed): GitHub UI → Sync fork.
Resolve any conflicts on a fresh branch, do NOT merge back.

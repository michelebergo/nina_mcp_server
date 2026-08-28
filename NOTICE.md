# Provenance and licensing

This repository is a fork of
[PaDev1/Nina_advanced_api_mcp](https://github.com/PaDev1/Nina_advanced_api_mcp).

**The upstream repository is MIT-licensed.** It carried no licence file until 28 August
2026, which under default copyright meant all rights reserved; asked about it, its author
confirmed the project was always intended as open source and MIT, and added the file the
same day.

## How much of this repository is upstream

Measured on 2026-08-28 against upstream commit `6ad09507ab6c0c06589c0b08314119c8880b8631`:

| | lines |
|---|---|
| upstream `nina_advanced_mcp.py` | 4811 |
| this repository's `nina_advanced_mcp.py` | 8161 |
| upstream lines still present here | 4806 (100% of upstream, 59% of this file) |

Counting code only, ignoring blank lines and comments, the figure is the same: 4019 of
4024. **This fork is the upstream file in full, plus roughly 3350 lines added here.** It is
not a rewrite that happens to share some structure.

## What the MIT licence in this repository does and does not cover

The `LICENSE` file covers the work added in this repository. It cannot by itself grant
rights over the upstream portions, because those rights were never granted to this
repository in the first place.

## Status

**Resolved.** The upstream author was asked directly and answered the same day:

> "The project is opensource and MIT. Sorry for missing adding the file. Will fix it."

The `LICENSE` file was added to the upstream repository eleven minutes later, and GitHub
now reports it as MIT. Both the written statement and the file stand as the record.

The MIT licence on this repository therefore rests on the upstream licence for the
inherited code and on its own author for the work added here.

- Request and answer: https://github.com/PaDev1/Nina_advanced_api_mcp/issues/5 (28 August 2026)
- Raised during: security review of `nina.plugin.opencode` by Stefan Berg, 2026-08-25
- Upstream author: [PaDev1](https://github.com/PaDev1) — the tool this one grew out of

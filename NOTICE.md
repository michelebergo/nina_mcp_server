# Provenance and licensing

This repository is a fork of
[PaDev1/Nina_advanced_api_mcp](https://github.com/PaDev1/Nina_advanced_api_mcp).

**The upstream repository carries no licence file.** Under default copyright that means
all rights are reserved by its author: forking it on GitHub is permitted by GitHub's terms
of service, but that permission does not extend to redistributing it elsewhere or to
placing a different licence on it.

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

Permission has been requested from the upstream author. Until that request is answered,
the licensing of the upstream portions of this repository is **unresolved**, and anyone
redistributing it should be aware of that.

- Request: https://github.com/PaDev1/Nina_advanced_api_mcp/issues/5 (opened 2026-08-28)
- Raised during: security review of `nina.plugin.opencode` by Stefan Berg, 2026-08-25

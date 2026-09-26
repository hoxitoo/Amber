"""Feature spec version — must match `version` in config/features.yaml.

Bumping it makes every per-symbol feature file recompute from scratch on the
next pipeline cycle (see `compute._resume_index`). That is what a new feature
column needs: incremental resume would otherwise append rows carrying the new
keys onto a file whose earlier rows lack them.

v5 (2026-09-26): forced-liquidation features from `allLiquidation`.
"""

FEATURE_SPEC_VERSION = "v5"

"""Research extensions for CAROTS.

This package contains everything added on top of the upstream CAROTS code base
for two research goals:

* Step 1 - stress-test CAROTS on three kinds of *flawed* VAR training data
  (no causal relationship, non-stationary causal relationship, anomalies in the
  training data). See ``datagen.py``.
* Step 2 - variable-level anomaly localization (added on the scoring side only,
  without touching training or the loss functions). See ``localization.py``.
  For a minimal demo with a clear linear root-cause chain, see
  ``datagen.py --variant toys``, ``TOY_CHAIN.md`` (the data) and
  ``LOCALIZATION_METRICS.md`` (how it is evaluated, and the controls).

The upstream code under ``CAROTS/`` is left untouched except for a few small,
clearly-marked hooks:

* ``config.py``          - ``DATA.VAR_DIR`` (point a run at a variant data
  directory, e.g. ``VAR_nocausal``), ``TEST.SAVE_PER_VARIABLE`` (opt-in to
  saving Step 2 artifacts) and ``CAUSAL_DISCOVERER_DIR`` (share one causal
  discoverer across the anomaly types of a scenario).
* ``datasets/build.py``  - ``VARSegLoader`` reads ``DATA.VAR_DIR``.
* ``models/carots/scorer_carots.py`` - ``get_per_variable_cd_scores`` returns
  the per-variable forecasting error used for Step 2.
* ``models/carots/predictor.py`` - saves ``per_variable_cd_error.npy`` and
  ``causality_matrix.npy`` when ``TEST.SAVE_PER_VARIABLE`` is set.
* ``models/carots/trainer_carots.py`` - loads/saves the causal discoverer from
  ``CAUSAL_DISCOVERER_DIR`` when provided.

Training, the augmentors, the SOC loss and the encoder are unchanged.
"""

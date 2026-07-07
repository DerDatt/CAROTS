"""Single source of truth for the Step 1 robustness experiments.

This module defines the experimental grid - which flawed-training-data
*scenarios* and which *anomaly types* we evaluate - and knows how to turn a
single cell of that grid into the exact ``main.py`` command line that runs
CAROTS on it. Keeping all of this in one place means the run scripts, the Colab
notebook and the results aggregator all agree on naming and paths.

Scenarios (Step 1)
------------------
- ``baseline``      : clean, stationary VAR with real causal structure (reference).
- ``nocausal``      : variables are autocorrelation-only, no cross-causal links.
- ``nonstationary`` : the causal relationships drift over time (regime switches).
- ``contaminated``  : the *training* data already contains anomalies.

Each scenario reads its data from ``data/VAR_<scenario>/`` (produced by
``experiments/datagen.py``) via the ``DATA.VAR_DIR`` config hook, and writes its
outputs to a scenario-specific ``RESULT_DIR`` so nothing collides.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------
# The four flawed-data scenarios, mapped to their data folders under data/.
SCENARIOS = {
    "baseline": "VAR_baseline",
    "nocausal": "VAR_nocausal",
    "nonstationary": "VAR_nonstationary",
    "contaminated": "VAR_contaminated",
}


@dataclass(frozen=True)
class AnomalySpec:
    """One anomaly type and the factor used when it was injected into the test set."""
    name: str          # e.g. "point_global"
    factor: str        # e.g. "2.0" or "None" (collective_global has no factor)

    @property
    def dataset_name(self) -> str:
        """The DATA.NAME CAROTS expects, e.g. VAR_point_global_factor2.0."""
        return f"VAR_{self.name}_factor{self.factor}"


# The four anomaly types from the paper. collective_global uses factor "None".
ANOMALIES = [
    AnomalySpec("point_global", "2.0"),
    AnomalySpec("point_contextual", "2.0"),
    AnomalySpec("collective_trend", "2.0"),
    AnomalySpec("collective_global", "None"),
]

# Default training hyper-parameters, matching the upstream scripts/VAR_*.sh.
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_SEEDS = [2]


@dataclass
class RunSpec:
    """A fully-resolved experiment: one (scenario, anomaly, seed) combination."""
    scenario: str
    anomaly: AnomalySpec
    seed: int
    base_dir: str = "data/"
    results_root: str = "results"
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    extra_opts: List[str] = field(default_factory=list)

    @property
    def var_dir(self) -> str:
        return SCENARIOS[self.scenario]

    @property
    def result_dir(self) -> str:
        """Scenario/seed-specific output root (DATA.NAME is appended by main.py)."""
        return os.path.join(self.results_root, self.scenario, f"seed{self.seed}")

    @property
    def causal_dir(self) -> str:
        """Shared causal-discoverer cache for this scenario/seed.

        This deliberately does NOT include the anomaly type, so all four anomaly
        types of a scenario reuse a single trained causal discoverer (they share
        the same train.npy). main.py does not append DATA.NAME to this key.
        """
        return os.path.join(self.results_root, self.scenario, f"seed{self.seed}",
                            "shared_causal")

    @property
    def data_dir(self) -> str:
        """Folder holding this scenario's *_varlabels.npy (for localization)."""
        return os.path.join(self.base_dir, self.var_dir)

    @property
    def final_result_dir(self) -> str:
        """Where main.py actually writes outputs after appending DATA.NAME."""
        return os.path.join(self.result_dir, self.anomaly.dataset_name)

    def to_opts(self, save_per_variable: bool = True) -> List[str]:
        """Build the yacs override list passed to ``main.py``."""
        opts = [
            "DATA.NAME", self.anomaly.dataset_name,
            "DATA.BASE_DIR", self.base_dir,
            "DATA.VAR_DIR", self.var_dir,
            "RESULT_DIR", self.result_dir,
            "TRAIN.CHECKPOINT_DIR", self.result_dir,
            "CAUSAL_DISCOVERER_DIR", self.causal_dir,
            "TEST.SAVE_PER_VARIABLE", "True" if save_per_variable else "False",
            "SOLVER.BASE_LR", str(self.lr),
            "SOLVER.WEIGHT_DECAY", str(self.weight_decay),
            "SEED", str(self.seed),
        ]
        opts += list(self.extra_opts)
        return opts

    def to_command(self, python: str = "python", save_per_variable: bool = True) -> str:
        """Render the full shell command for this run."""
        return f"{python} main.py " + " ".join(self.to_opts(save_per_variable))


def iter_runs(scenarios: Optional[List[str]] = None,
              anomalies: Optional[List[AnomalySpec]] = None,
              seeds: Optional[List[int]] = None,
              results_root: str = "results",
              base_dir: str = "data/") -> List[RunSpec]:
    """Enumerate every (scenario, anomaly, seed) run in the requested grid."""
    scenarios = scenarios or list(SCENARIOS.keys())
    anomalies = anomalies or ANOMALIES
    seeds = seeds or DEFAULT_SEEDS
    runs = []
    for scenario in scenarios:
        for anomaly in anomalies:
            for seed in seeds:
                runs.append(RunSpec(
                    scenario=scenario, anomaly=anomaly, seed=seed,
                    results_root=results_root, base_dir=base_dir,
                ))
    return runs

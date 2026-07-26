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

Step 2 helper scenarios
-----------------------
- ``toy_chain``      : small linear causal chain ``0 -> 1 -> 2`` plus distractors,
  with test anomalies on the chain **root** (variable 0).
- ``toy_chain_mid``  : same process, anomalies on the **middle** of the chain
  (variable 1). Control for "does the attribution just always answer 0?" and the
  case where causal propagation should pull credit back from the child.
- ``toy_chain_leaf`` : same process, anomalies on the chain **leaf** (variable 2).
  The leaf has no children, so propagation has no evidence to gather; if the
  propagated score is worse than the direct one here, that is the cost of the
  propagation term.

None of these are part of the default Step-1 grid. See
``experiments/TOY_CHAIN.md`` for the generative model and
``experiments/LOCALIZATION_METRICS.md`` for how they are evaluated.

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
# Step-1 flawed-data scenarios (default for ``iter_runs`` / ``run_all``).
STEP1_SCENARIOS = ("baseline", "nocausal", "nonstationary", "contaminated")

# Step-2 toy scenarios: same generative process, different injection target.
# The value is the variable that receives the test anomalies, i.e. the single
# correct localization answer. Consumed by experiments/datagen.py via
# ``--toy-root-var`` (see generate_toy_scenarios in run_pipeline.sh).
TOY_SCENARIOS = {
    "toy_chain": 0,       # chain root
    "toy_chain_mid": 1,   # middle of the chain
    "toy_chain_leaf": 2,  # chain leaf (no causal children)
}

# All known scenario → data-folder mappings (includes the Step-2 toys).
SCENARIOS = {
    "baseline": "VAR_baseline",
    "nocausal": "VAR_nocausal",
    "nonstationary": "VAR_nonstationary",
    "contaminated": "VAR_contaminated",
    **{name: f"VAR_{name}" for name in TOY_SCENARIOS},
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

# Toy-chain uses a stronger factor; datagen writes factor-3.0 files by default.
TOY_ANOMALIES = [
    AnomalySpec("point_global", "3.0"),
    AnomalySpec("point_contextual", "3.0"),
    AnomalySpec("collective_trend", "3.0"),
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
    """Enumerate every (scenario, anomaly, seed) run in the requested grid.

    Defaults to the four Step-1 scenarios (the toys are opt-in via e.g.
    ``scenarios=["toy_chain"]``). When only toy scenarios are requested and no
    anomaly list is passed, ``TOY_ANOMALIES`` (factor 3.0) is used.
    """
    scenarios = scenarios or list(STEP1_SCENARIOS)
    if anomalies is None:
        only_toys = set(scenarios) <= set(TOY_SCENARIOS)
        anomalies = TOY_ANOMALIES if only_toys else ANOMALIES
    seeds = seeds or DEFAULT_SEEDS
    runs = []
    for scenario in scenarios:
        if scenario not in SCENARIOS:
            raise KeyError(
                f"Unknown scenario {scenario!r}; expected one of {list(SCENARIOS)}"
            )
        for anomaly in anomalies:
            for seed in seeds:
                runs.append(RunSpec(
                    scenario=scenario, anomaly=anomaly, seed=seed,
                    results_root=results_root, base_dir=base_dir,
                ))
    return runs

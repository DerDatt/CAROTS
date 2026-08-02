"""Generate (and optionally execute) the Step 1 experiment commands.

Training CAROTS needs a CUDA GPU (the model hard-codes ``.cuda()``), so the
normal workflow is:

    1. Locally: generate the data variants (experiments/datagen.py) and the run
       scripts (this file).
    2. On Colab/GPU: execute the generated scripts to train + evaluate.
    3. Locally: aggregate the results (experiments/aggregate.py).

This script writes:
    experiments/generated/run_<scenario>.sh   one script per scenario
    experiments/generated/run_all.sh          master script that calls them all
    experiments/generated/commands.txt        every command, one per line

Use ``--execute`` to run the commands here as well (only useful on a GPU box).
"""

import argparse
import os
import stat
import subprocess
import sys

# Allow running both as a module (python -m experiments.run_experiments) and as a
# plain script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.scenarios import (  # noqa: E402
    ANOMALIES, DEFAULT_SEEDS, SCENARIOS, STEP1_SCENARIOS, TOY_ANOMALIES,
    TOY_SCENARIOS, iter_runs,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_GEN_DIR = os.path.join(_HERE, "generated")


def _write_script(path, lines, keep_going=False):
    """Write an executable shell script with a header and the given lines.

    With ``keep_going`` the script drops ``set -e`` and instead reports how many
    commands failed. This matters for unattended runs: a single cell that dies
    (an OOM, a bad seed) must not throw away the rest of the night.
    """
    with open(path, "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -uo pipefail\n" if keep_going else "set -euo pipefail\n")
        # Run from the repository root so relative data/ and results/ paths work.
        f.write('cd "$(dirname "$0")/../.."\n\n')
        if keep_going:
            f.write("_failed=0\n")
            f.write("_run() { \"$@\" || { _failed=$((_failed+1)); "
                    "echo \"FAILED: $*\" >&2; }; }\n\n")
        for line in lines:
            f.write(line + "\n")
        if keep_going:
            f.write('\necho "$(basename "$0"): $_failed command(s) failed"\n')
            f.write("exit 0\n")
    # chmod +x
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main():
    parser = argparse.ArgumentParser(description="Generate Step 1 run scripts.")
    parser.add_argument("--scenarios", nargs="*", default=list(STEP1_SCENARIOS),
                        choices=list(SCENARIOS.keys()),
                        help="Subset of scenarios to generate "
                             "(default: Step-1 four; pass toy_chain for the "
                             "localization demo).")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS,
                        help="Random seeds to run for each cell.")
    parser.add_argument("--results-root", default="results",
                        help="Root folder for outputs (DATA.NAME appended by main.py).")
    parser.add_argument("--python", default="python",
                        help="Python executable to invoke in the generated commands.")
    parser.add_argument("--no-per-variable", action="store_true",
                        help="Disable saving the Step 2 per-variable localization artifacts.")
    parser.add_argument("--execute", action="store_true",
                        help="Also execute the commands now (requires a GPU).")
    parser.add_argument("--keep-going", action="store_true",
                        help="Generated scripts continue after a failing run "
                             "and report the failure count at the end. Use this "
                             "for unattended overnight runs.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Omit runs whose results already contain test.txt. "
                             "Lets you fill gaps without retraining finished "
                             "cells.")
    args = parser.parse_args()

    os.makedirs(_GEN_DIR, exist_ok=True)
    save_per_variable = not args.no_per_variable

    all_commands = []
    master_lines = []
    for scenario in args.scenarios:
        # Toy data is generated with factor 3.0 by default; Step-1 VARs use 2.0.
        anomaly_grid = TOY_ANOMALIES if scenario in TOY_SCENARIOS else ANOMALIES
        runs = iter_runs(scenarios=[scenario], anomalies=anomaly_grid,
                         seeds=args.seeds, results_root=args.results_root)
        if args.skip_existing:
            keep = [r for r in runs
                    if not os.path.exists(os.path.join(r.final_result_dir,
                                                       "test.txt"))]
            if len(keep) != len(runs):
                print(f"  {scenario}: skipping {len(runs) - len(keep)} finished "
                      f"run(s)")
            runs = keep
        cmds = [r.to_command(python=args.python, save_per_variable=save_per_variable)
                for r in runs]
        all_commands.extend(cmds)

        script_path = os.path.join(_GEN_DIR, f"run_{scenario}.sh")
        body = [f'_run {c}' for c in cmds] if args.keep_going else list(cmds)
        _write_script(script_path, [f'echo "=== {scenario} ==="'] + body,
                      keep_going=args.keep_going)
        master_lines.append(f'bash "$(dirname "$0")/run_{scenario}.sh"')
        print(f"wrote {script_path} ({len(cmds)} runs)")

    _write_script(os.path.join(_GEN_DIR, "run_all.sh"), master_lines,
                  keep_going=args.keep_going)
    with open(os.path.join(_GEN_DIR, "commands.txt"), "w") as f:
        f.write("\n".join(all_commands) + "\n")
    print(f"wrote {os.path.join(_GEN_DIR, 'run_all.sh')} and commands.txt "
          f"({len(all_commands)} total runs)")

    if args.execute:
        repo_root = os.path.dirname(_HERE)
        for cmd in all_commands:
            print(f"\n>>> {cmd}")
            subprocess.run(cmd, shell=True, check=True, cwd=repo_root)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Seed sweeps over ground station count for SCORE and Powell.
#
# Run from the repository root with the project environment active:
#
#   bash scripts/run_sweeps.sh score_capella   # SCORE, 5 CAPELLA satellites, seeds 0-4
#   bash scripts/run_sweeps.sh score_iceye     # SCORE, 34 ICEYE satellites, seeds 0-4
#   bash scripts/run_sweeps.sh powell          # Powell, 5 CAPELLA satellites, seeds 5-10
#
# Extra arguments are passed to every run as Hydra overrides, e.g. debug.wandb=True.

GS_NUMS="1 2 3 4 5 7 10 15 20"

sweep() {
  # sweep "<seeds>" <hydra overrides...>
  local seeds=$1; shift
  for gs_num in $GS_NUMS; do
    for randseed in $seeds; do
      python src/main.py problem.gs_num=$gs_num debug.randseed=$randseed "$@"
    done
  done
}

case "$1" in
  score_capella) shift; sweep "0 1 2 3 4" problem.method=nelder_ccgs "$@" ;;
  score_iceye)   shift; sweep "0 1 2 3 4" problem.method=nelder_ccgs problem.sat_num=34 scenario.constellations=ICEYE "$@" ;;
  powell)        shift; sweep "5 6 7 8 9 10" problem.method=powell "$@" ;;
  *) echo "usage: bash scripts/run_sweeps.sh {score_capella|score_iceye|powell} [hydra overrides...]"; exit 1 ;;
esac

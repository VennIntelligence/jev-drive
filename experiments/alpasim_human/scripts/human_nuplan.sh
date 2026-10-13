#!/usr/bin/env bash
# nuPlan track by hand (phone wheel), Tokyo card 0: human_nuplan.sh <out dir> <scene id file> [wizard overrides]. Env: human_stack.sh.
exec bash "$(dirname "$0")/human_stack.sh" nuplan "$@"

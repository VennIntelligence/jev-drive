#!/usr/bin/env bash
# PAI track by hand (phone wheel), Tokyo card 0: human_pai.sh <out dir> <scenes.tsv | scene id file> [wizard overrides]. Env: human_stack.sh.
exec bash "$(dirname "$0")/human_stack.sh" pai "$@"

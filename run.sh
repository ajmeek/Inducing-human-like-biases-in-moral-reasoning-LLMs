#!/bin/bash 
# 
# Script to run high level functions for development, maintenance, deployment, etc.

USAGE=`cat<<EOF
Usage: run.sh <function> [parameters...]

FUNCTIONS: 
  install           - installs environment to load data, train
  train             - runs training
  datasets          - downloads and processes a dataset(s)
EOF
`
set -euo pipefail
IFS=$'\n\t'
trap "echo 'error: Script failed: see failed command above'" ERR

# Useful paths:
readonly script_path="${BASH_SOURCE[0]}"
script_dir="$(dirname "$script_path")"
readonly script_dir
root_dir=$( realpath "$script_dir" )

export GIT_MAIN_BRANCH_NAME=main
export GIT_REMOTE=github.com/ameek2/Inducing-human-like-biases-in-moral-reasoning-LLMs.git
export AISCBB_ARTIFACTS_DIR=${AISCBB_ARTIFACTS_DIR:-$root_dir/artifacts}
export AISCBB_DATA_DIR=${AISCBB_DATA_DIR:-$root_dir/data}

################################################################################

function provision() {
    python3 -m pip install pipenv 
    pipenv --python 3.10 install
    python3 -m pip install datalad-installer
    datalad-installer --sudo ok git-annex -m datalad/git-annex:release
}

function datasets() {
    source ./bin/_datasets.sh
}

function gcp() {
    source ./bin/_gcp.sh
}

function train() {
    pipenv run python3 "$root_dir/src/main.py" "$@"
}

################################################################################


if [[ $# == 0 ]] ; then 
    echo "$USAGE"
else 
    cmd=$1
    shift 1
    $cmd "$@"  
fi

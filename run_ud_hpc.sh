#!/bin/bash 
# 
# Script to run high level functions for development, maintenance, deployment, etc.

# Modified version to use on UD HPC cluster.

USAGE=`cat ../../README.md`

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
export CURRENT_GIT_BRANCH=$( [[ -e ./.git ]] && git rev-parse --abbrev-ref HEAD )
export PYTHON_ENV_NAME=brainbias

################################################################################


function datasets() {
    echo cwd: $(pwd)
    ./data/ds000212/make.sh "$@"
}

function train() {
    python3 "$root_dir/src/main.py" "$@"
}

function _mamba() {
    if ! which mamba ; then
        wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
        bash Mambaforge-$(uname)-$(uname -m).sh
        rm  Mambaforge-$(uname)-$(uname -m).sh
        echo "Please re-enter into your shell to activate Mamba. Then run again ./run.sh vast"
    fi
}

function local() {
    _mamba
    mamba env create --force -n $PYTHON_ENV_NAME -f environment-cpu.yml
    mamba activate $PYTHON_ENV_NAME
    echo Done
}

function vast() {
    source bin/_vast.sh "$@"
}

function gcp() {
    source bin/_gcp.sh "$@"
}

##########################################################################


if [[ $# == 0 ]] ; then 
    echo "$USAGE"
else 
    cmd=$1
    shift 1
    $cmd "$@"  
fi

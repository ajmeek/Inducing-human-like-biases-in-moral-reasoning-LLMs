#!/bin/bash 
# 
# Script to run high level functions for development, maintenance, deployment, etc.

USAGE=`cat README.md`

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
    ./data/ds000212/make.sh "$@"
}

function train() {
    python3 "$root_dir/src/main.py" "$@"
}

function _mamba() {
    export MAMBA_SH="${HOME}/mambaforge/etc/profile.d/conda.sh"
    if [[ ! -e $MAMBA_SH ]] ; then
        echo 'Installing Mamba...'
        wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
        bash Mambaforge-$(uname)-$(uname -m).sh
        rm  Mambaforge-$(uname)-$(uname -m).sh
    fi
    source $MAMBA_SH 
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

function test() {
    (cd data/ds000212/ds000212_lfb ; pytest )
}

##########################################################################


if [[ $# == 0 ]] ; then 
    echo "$USAGE"
else 
    cmd=$1
    shift 1
    $cmd "$@"  
fi

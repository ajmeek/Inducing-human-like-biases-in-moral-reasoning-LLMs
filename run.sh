#!/bin/bash 
# 
# Script to run high level functions for development, maintenance, deployment, etc.

USAGE=`cat<<EOF
Usage: run.sh ( <function>... | <function> [parameters...] )

FUNCTIONS: 
  install           - installs environment to load data, train
  train             - runs training
  prepare_datasets  - downloads and processes a dataset(s)
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


function install() {
    echo Installing...
    git config --global user.email "artyomkarpov@gmail.com"
    git config --global user.name "Artem K"
    python3 -m pip install -r requirements.txt
    python3 -m pip install datalad-installer
    datalad-installer --sudo ok datalad git-annex -m datalad/git-annex:release 
}

function prepare_datasets() {
    echo Preparing datasets...
    source ./bin/datasets.sh
}

function train() {
    echo Training...
    bash ./bin/train.sh "$@"
}

if [[ $# == 0 ]] ; then 
    echo "$USAGE"
else 
    if [[ "$@" =~ '--' ]]; then 
        cmd=$1
        shift 1
        $cmd "$@"
    else
        while [[ $# -ne 0 ]] ; do 
            $1
            shift 1
        done
    fi
fi


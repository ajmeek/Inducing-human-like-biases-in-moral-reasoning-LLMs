#!/bin/bash 
set -euo pipefail
IFS=$'\n\t'
trap "echo 'error: Script failed: see failed command above'" ERR

# Useful paths:
readonly script_path="${BASH_SOURCE[0]}"
script_dir="$(dirname "$script_path")"
readonly script_dir
root_dir=$( realpath "$script_dir/.." )


function prepare_datasets() 
{
    source ./bin/datasets.sh
}

function train()
{
    load_data
    echo Training
    bash bin/train.sh --num_epochs=0 --only_train_head=True  --num_samples_test=1000 --num_samples_train=3000 --ethics_num_epochs=1
}

function install()
{

    git config --global user.email "artyomkarpov@gmail.com"
    git config --global user.name "Artem K"
    python3 -m pip install -r requirements.txt
    python3 -m pip install datalad-installer
    datalad-installer --sudo ok datalad git-annex -m datalad/git-annex:release 
}



if [[ $# == 0 ]] ; then 
    echo 'Usage: run.sh <FUNCTIONS...>

FUNCTIONS: 
  install   - installs environment to load data, train
  train     - runs training
  prepare_datasets - downloads and processes a dataset(s)
'
else 
    cd $root_dir 
    while [[ $# -ne 0 ]] ; do 
        $1
        shift 1
    done
fi



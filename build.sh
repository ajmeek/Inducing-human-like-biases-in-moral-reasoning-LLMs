#!/bin/bash
set -euo pipefail
IFS=$'\n\t'
trap "echo 'error: Script failed: see failed command above'" ERR

# Useful paths:
readonly script_path="${BASH_SOURCE[0]}"
script_dir="$(dirname "$script_path")"
readonly script_dir
root_dir=$( realpath "$script_dir/.." )

if [[ ! -e data/ds000212  || ! -e data/functional_flattened ]]; then 
    echo Downloading and processing ds000212...
    python bin/fMRI_utils.py
fi

if [[ ! -e data/ethics ]]; then 
    echo Downloading and processing ETHICS...
    bash ./bin/download_ethics_ds.sh 
fi

echo Training
bash bin/train.sh --num_epochs=0 --only_train_head=True  --num_samples_test=1000 --num_samples_train=3000 --ethics_num_epochs=1
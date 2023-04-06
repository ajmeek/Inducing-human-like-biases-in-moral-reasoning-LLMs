#!/bin/bash

: '
Run the script below to download code and install dependances.
# Token from https://github.com/settings/tokens
YOURTOKEN=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  
git clone "https://$YOURTOKEN@github.com/ameek2/Inducing-human-like-biases-in-moral-reasoning-LLMs.git"
cd Inducing-human-like-biases-in-moral-reasoning-LLMs
git checkout ds000212_fine_tuning_bert
pip install -r requirements.txt
'

set -euo pipefail
IFS=$'\n\t'
trap "echo 'error: Script failed: see failed command above'" ERR

# Useful paths:
readonly script_path="${BASH_SOURCE[0]}"
script_dir="$(dirname "$script_path")"
readonly script_dir
root_dir=$( realpath "$script_dir/.." )


if [[ ! -e data/functional_flattened ]]; then 
    echo Downloading and processing ds000212...
    python bin/fMRI_utils.py
fi

if [[ ! -e data/ethics ]]; then 
    echo Downloading and processing ETHICS...
    bash ./bin/download_ethics_ds.sh 
fi

echo Training
bash bin/train.sh --num_epochs=0 --only_train_head=True  --num_samples_test=1000 --num_samples_train=3000 --ethics_num_epochs=1
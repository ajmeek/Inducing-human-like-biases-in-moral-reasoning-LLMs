#!/bin/bash
set -euo pipefail
IFS=$'\n\t'
trap "echo 'error: Script failed: see failed command above'" ERR
readonly script_path="${BASH_SOURCE[0]}"
script_dir="$(dirname "$script_path")"
readonly script_dir
root_dir=$( realpath "$script_dir/.." )

[[ -e "$root_dir/data" ]] || ( echo 'data dir not found'  ; exit 1 )

# Download and extract:
pushd "$root_dir/data"
[[ -e ethics.tar ]] || curl -O 'https://people.eecs.berkeley.edu/~hendrycks/ethics.tar' 
mkdir -p ethics
tar -xvf ethics.tar 
rm ./ethics.tar
popd

[[ -e "$root_dir/data/ethics/commonsense/cm_train.csv" ]] || ( echo 'downloading ethics ds failed'  ; exit 1 )
echo 'done'
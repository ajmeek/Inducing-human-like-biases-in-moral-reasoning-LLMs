#!/bin/bash
set -euo pipefail
IFS=$'\n\t'
trap "echo 'error: Script failed: see failed command above'" ERR

# Useful paths:
readonly script_path="${BASH_SOURCE[0]}"
script_dir="$(dirname "$script_path")"
readonly script_dir
root_dir=$( realpath "$script_dir/.." )

# Run:
pushd $root_dir > /dev/null
python3 "$root_dir/src/main.py" "$@"  || popd
popd
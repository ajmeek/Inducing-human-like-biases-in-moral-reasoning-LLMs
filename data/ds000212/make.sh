#!/bin/bash 

USAGE="
Script to preprocess ds000212 dataset. Available datasets:

1. ds000212_raw;
2. ds000212_roi;
3. ds000212_fmriprep.
"

set -euo pipefail
IFS=$'\n\t'
trap "echo 'error: Script failed: see failed command above'" ERR

# Useful paths:
readonly script_path="${BASH_SOURCE[0]}"
script_dir="$(dirname "$script_path")"
readonly script_dir
base_dir=$( realpath "$script_dir" )

if [[ ! -v num_cpus ]] ; then
num_cpus=$( python -c 'import multiprocessing; print(max(1,  multiprocessing.cpu_count() - 2))' )
fi 

function ds000212_scenarios() {
    if [[ ! -e $base_dir/ds000212_scenarios.csv ]]; then
        cp $base_dir/ds000212_scenarios.csv $base_dir/ds000212_scenarios.csv
    fi
}

function _ds000212_download() {
    if [[ ! -e $base_dir/ds000212 ]] ; then
        echo Downloading ds000212...
        datalad install -s https://github.com/OpenNeuroDatasets/ds000212.git $base_dir/ds000212
    fi
}

function ds000212_raw() {
    _ds000212_download
    ds000212_scenarios
    export TARGET_DS_NAME=ds000212_raw
    export SOURCE_DS_NAME=ds000212
    export IS_ROI_ARG=--no-roi
    echo Running make for $TARGET_DS_NAME
    (cd $base_dir ; make -f $base_dir/ds000212.mk all --jobs $num_cpus --silent )
}

function ds000212_roi() {
    _ds000212_download
    ds000212_scenarios
    export TARGET_DS_NAME=ds000212_roi
    export SOURCE_DS_NAME=ds000212
    export IS_ROI_ARG=--roi
    echo Running make for $TARGET_DS_NAME
    (cd $base_dir ; make -f $base_dir/ds000212.mk all  --jobs $num_cpus  --silent )
}

function ds000212_fmriprep() {
    _ds000212_download
    ds000212_scenarios
    export TARGET_DS_NAME=ds000212_fmriprep
    export SOURCE_DS_NAME=ds000212/derivatives/ds000212-fmriprep
    if [[ ! -e $base_dir/$SOURCE_DS_NAME ]] ; then
        echo Downloading $base_dir/$SOURCE_DS_NAME...
        datalad install -s https://github.com/OpenNeuroDerivatives/ds000212-fmriprep.git "$base_dir/$SOURCE_DS_NAME"
    fi
    echo Running make for $TARGET_DS_NAME
    (cd $base_dir ; make -f $base_dir/ds000212prep.mk all  --jobs 1 )
}


if [[ $# == 0 || "$0" == '-h' || "$0" == '--help' ]]; then 
    echo $USAGE
    exit 1
fi
while [[ $# != 0 ]] ; do
    cmd=$1
    $cmd
    shift 1
done


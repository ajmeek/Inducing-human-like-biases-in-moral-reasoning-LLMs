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

GIT_MAIN_BRANCH_NAME=main
GIT_REMOTE=github.com/ameek2/Inducing-human-like-biases-in-moral-reasoning-LLMs.git

################################################################################

function datasets() {
    echo Preparing datasets...
    source ./bin/datasets.sh
}

function train() {
    echo Training...
    bash ./bin/train.sh "$@"
}

function gcp() {
    GCPUSAGE="Provide task after gcp like this: run.sh gcp my-task [parameter ...]"

    # If it is local or remote environment:
    if [[ -z ${AISCIBB_GCP_FLAG-} ]] ; then
        # If all prerequisits met:
        if [[ $# -eq 0 ]]; then
            echo Failed to run at GCP: \n $GCPUSAGE
            exit 1
        fi
        [[ ! -z ${AISCIBB_GIT_TOKEN-} ]]  || ( echo "Please set AISCIBB_GIT_TOKEN environment variable (see https://github.com/settings/tokens)." ; exit 1 )
        [[ ! -z ${AISCIBB_GCP_SSH_USERHOST-} ]]  || ( echo "Please set AISCIBB_GCP_SSH_USERHOST environment variable (example: user@123.123.123.123)." ; exit 1 )

        echo Deploying to GCP...
        scp -q ~/.gitconfig scp://$AISCIBB_GCP_SSH_USERHOST/.gitconfig
        scp -q ./run.sh scp://$AISCIBB_GCP_SSH_USERHOST/run.sh
        # Run remotely:
        ssh ssh://$AISCIBB_GCP_SSH_USERHOST 'AISCIBB_GCP_FLAG=1 bash' ./run.sh gcp $( git rev-parse --abbrev-ref HEAD ) "$AISCIBB_GIT_TOKEN" "$@"
    else
        # If all prerequisits met:
        if [[ $# -le 2 ]]; then
            echo Failed to run at GCP: $GCPUSAGE
            exit 1
        fi

        echo At GCP. Running deployment...
        
        echo Retrieving files...
        GITBRANCH=${1-$GIT_MAIN_BRANCH_NAME}
        AISCIBB_GIT_TOKEN=$2

        TARGETDIR=~/aiscbbproj
        if [[ -e $TARGETDIR ]] ; then 
            echo Removing old directory
            rm -rf $TARGETDIR
        fi

        git clone -b $GITBRANCH "https://$AISCIBB_GIT_TOKEN@$GIT_REMOTE" $TARGETDIR 
        cd $TARGETDIR

        echo Building docker image...
        docker buildx build -t aiscbb $TARGETDIR

        echo Running container...
        shift 2  # Remove first two params for gcp.
        mkdir -p ~/artifacts
        mkdir -p ~/data
        docker container run \
            -e AISCBB_ARTIFACTS_DIR=/asicbb_data \
            -e AISCBB_DATA_DIR=/aiscbb_artifacts \
            -v ~/artifacts:/aiscbb_artifacts \
            -v ~/data:/asicbb_data \
            -v ~/.gitconfig:/etc/gitconfig \
            aiscbb \
            bash run.sh "$@"

        [[ ! -e %TARGETDIR ]] || rm -dr $TARGETDIR
        echo At GCP. Finished.
    fi
}

################################################################################


if [[ $# == 0 ]] ; then 
    echo "$USAGE"
else 
    cmd=$1
    shift 1
    $cmd "$@"
fi


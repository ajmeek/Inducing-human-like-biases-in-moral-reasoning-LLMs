#!/bin/bash 
# 
# Script to run high level functions for development, maintenance, deployment, etc.

USAGE=`cat<<EOF
Usage: run.sh ( <function>... | <function> [parameters...] )

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

GIT_REMOTE=git@github.com:ameek2/Inducing-human-like-biases-in-moral-reasoning-LLMs.git
GIT_MAIN_BRANCH_NAME=main
AISCIBB_GIT_TOKEN=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  

################################################################################

function install() {
    echo Installing...
    python3 -m pip install -r requirements.txt
    python3 -m pip install datalad-installer
    datalad-installer --sudo ok datalad git-annex -m datalad/git-annex:release 
}

function datasets() {
    echo Preparing datasets...
    source ./bin/datasets.sh
}

function train() {
    echo Training...
    bash ./bin/train.sh "$@"
}

function gcp() {
    if ( printenv AISCIBB_GCP_FLAG > /dev/null ) ; then

        echo At GCP. Running deployment...

        echo Retrieving files.
        GITBRANCH=${1-$GIT_MAIN_BRANCH_NAME}
        git clone -b $GITBRANCH $GIT_REMOTE proj

        git clone "https://$YOURTOKEN@github.com/ameek2/Inducing-human-like-biases-in-moral-reasoning-LLMs.git"
        cd proj
        
        echo bash ./run.sh install 

        echo bash ./run.sh datasets 

    else
        echo Deploying to GCP...
        [[ "$AISCIBB_GCP_SSH_USERHOST" != "" ]] || ( echo 'Need SSH URL (AISCIBB_GCP_SSH_DESTINATION variable).' ; exit 1 )
        scp -q ./run.sh scp://$AISCIBB_GCP_SSH_USERHOST/run.sh
        ssh ssh://$AISCIBB_GCP_SSH_USERHOST 'AISCIBB_GCP_FLAG=1 bash' ./run.sh gcp $( git rev-parse --abbrev-ref HEAD )
    fi
}


################################################################################


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


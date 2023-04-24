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
DEPLOYLOCKFILENAME=AISCBB_proj.lock

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

    if [[ -z ${AISCIBB_GCP_FLAG-} ]] ; then
        [[ ! -z ${AISCIBB_GIT_TOKEN-} ]]  || ( echo Please set AISCIBB_GIT_TOKEN environment variable  ; exit 1 )
        [[ ! -z ${AISCIBB_GCP_SSH_USERHOST-} ]]  || ( echo Please set AISCIBB_GCP_SSH_USERHOST environment variable  ; exit 1 )

        echo Deploying to GCP...
        [[ "$AISCIBB_GCP_SSH_USERHOST" != "" ]] || ( echo 'Need SSH URL (AISCIBB_GCP_SSH_DESTINATION variable).' ; exit 1 )
        scp -q ./run.sh scp://$AISCIBB_GCP_SSH_USERHOST/run.sh
        ssh ssh://$AISCIBB_GCP_SSH_USERHOST 'AISCIBB_GCP_FLAG=1 bash' ./run.sh gcp $( git rev-parse --abbrev-ref HEAD ) "$AISCIBB_GIT_TOKEN"
    else
        echo At GCP. Running deployment...
        
        echo Locking...
        LOCK=/tmp/$DEPLOYLOCKFILENAME
        [[ ! -e $LOCK ]] || ( echo "Failed to deploy: remote machine is locked (file $LOCK). Last access:"; stat $LOCK | grep Access ; exit 1 )
        touch $LOCK

        echo Retrieving files...
        GITBRANCH=${1-$GIT_MAIN_BRANCH_NAME}
        AISCIBB_GIT_TOKEN=$2

        TARGETDIR=~/aiscbbproj
        [[ ! -e %TARGETDIR ]] || rm -dr $TARGETDIR
        git clone -b $GITBRANCH "https://$AISCIBB_GIT_TOKEN@$GIT_REMOTE" $TARGETDIR 
        cd $TARGETDIR
        
        bash ./run.sh install  || ( echo 'Failed to install' ; exit 1)

        # bash ./run.sh datasets  || ( echo 'Failed to install' ; exit 1)

        [[ ! -e %TARGETDIR ]] || rm -dr $TARGETDIR
        rm $LOCK
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


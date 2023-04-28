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
AISCBB_ARTIFACTS_DIR=${AISCBB_ARTIFACTS_DIR:-$root_dir/artifacts}
AISCBB_DATA_DIR=${AISCBB_DATA_DIR:-$root_dir/data}

################################################################################

function datasets() {
    echo Preparing datasets...
    source ./bin/datasets.sh
}

function train() {
    echo Training...
    bash ./bin/train.sh "$@"
}

# TODO: refactor, move to its file.
function gcp() {
    cat - <<END 
        Runs tasks at Google Cloud Platform. 
        Provide task after gcp like this: run.sh gcp my-task [parameter ...].  
        If no tasks provided (run.sh gcp) it syncs remote and local files (gets results)."
END

    if [[ -z ${AISCIBB_GCP_FLAG-} ]] ; then
        # In local environment.
        [[ ! -z ${AISCIBB_GIT_TOKEN-} ]]  || ( echo "Please set AISCIBB_GIT_TOKEN environment variable (see https://github.com/settings/tokens)." ; exit 1 )
        [[ ! -z ${AISCIBB_GCP_SSH_USERHOST-} ]]  || ( echo "Please set AISCIBB_GCP_SSH_USERHOST environment variable (example: user@123.123.123.123)." ; exit 1 )

        if [[ $# -gt 0 ]]; then
            echo Calling GCP to run a task...
            scp -q ~/.gitconfig scp://$AISCIBB_GCP_SSH_USERHOST/.gitconfig
            scp -q ./run.sh scp://$AISCIBB_GCP_SSH_USERHOST/run.sh
            ssh ssh://$AISCIBB_GCP_SSH_USERHOST 'AISCIBB_GCP_FLAG=1 bash' ./run.sh gcp $( git rev-parse --abbrev-ref HEAD ) "$AISCIBB_GIT_TOKEN" "$@"
        fi

        echo Getting results
        ssh ssh://$AISCIBB_GCP_SSH_USERHOST 'AISCIBB_GCP_FLAG=1 bash' ./run.sh gcp

        echo Syncing artifacts from GCP...
        # TODO: configure artifacts dir at remote.
        rsync -rP $AISCIBB_GCP_SSH_USERHOST:~/artifacts $AISCBB_ARTIFACTS_DIR
        echo Done. Artifacts at $AISCBB_ARTIFACTS_DIR
    else
        echo At GCP.

        # In remote environment.
        # If all prerequisits met:
        mkdir -vp $AISCBB_ARTIFACTS_DIR
        mkdir -vp $AISCBB_DATA_DIR

        C_ID_FILE=./aiscbb_container_id
        if [[ $# -eq 0 ]]; then
            echo Reporting progress...
            sudo docker stats --no-stream 
            if [[ -e $C_ID_FILE ]]; then
                C_ID=$( cat)
                sudo docker top $C_ID ps -x
                C_LOG_FILE="$AISCBB_ARTIFACTS_DIR/$( date +%Y-%m-%d-%H%M )_run_sh.log "
                docker container logs $C_ID 2> $C_LOG_FILE
            fi
        else
            echo Run a task...
            echo Stoping current task if any.
            if [[ -e $C_ID_FILE && $( docker stats --no-stream $( cat $C_ID_FILE ) ) ]]; then
                C_ID=$( cat $C_ID_FILE )
                echo There is container running: $( docker top $C_ID ps -x ). Stoppping...
                docker stop $C_ID
                rm $C_ID_FILE
            fi
            
            echo Clonning repository...
            GITBRANCH=${1-$GIT_MAIN_BRANCH_NAME}
            AISCIBB_GIT_TOKEN=$2
            TARGETDIR=~/aiscbbproj
            if [[ -e $TARGETDIR ]] ; then 
                echo Removing old repo directory
                rm -rf $TARGETDIR
            fi
            git clone -b $GITBRANCH "https://$AISCIBB_GIT_TOKEN@$GIT_REMOTE" $TARGETDIR 

            echo Building docker image...
            ( cd $TARGETDIR ; docker buildx build -t aiscbb . )

            echo Running container...
            shift 2  # Remove first two params for gcp.
            >$C_ID_FILE docker container run \
                -e AISCBB_ARTIFACTS_DIR=/aiscbb_artifacts \
                -e AISCBB_DATA_DIR=/asicbb_data \
                -v $AISCBB_ARTIFACTS_DIR:/aiscbb_artifacts \
                -v $AISCBB_DATA_DIR:/asicbb_data \
                -v ~/.gitconfig:/etc/gitconfig \
                --detach \
                aiscbb \
                bash run.sh "$@" 
            C_ID=$( cat $C_ID_FILE )

            docker container attach $C_ID
            
            C_LOG_FILE="$AISCBB_ARTIFACTS_DIR/$( date +%Y-%m-%d-%H% )_run_sh.log "
            docker container logs $C_ID 2> $C_LOG_FILE

            [[ ! -e %TARGETDIR ]] || rm -dr $TARGETDIR
            echo At GCP. Finished.
        fi
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


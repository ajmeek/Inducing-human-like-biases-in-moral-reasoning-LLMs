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
#root_dir=$(readlink "$script_dir" )
cd "$script_dir" || exit 1
root_dir="$(pwd)"

export GIT_MAIN_BRANCH_NAME=main
export GIT_REMOTE=github.com/ameek2/Inducing-human-like-biases-in-moral-reasoning-LLMs.git
export AISCBB_ARTIFACTS_DIR=${AISCBB_ARTIFACTS_DIR:-$root_dir/artifacts}
export AISCBB_DATA_DIR=${AISCBB_DATA_DIR:-$root_dir/data}

################################################################################

function provision() {
    python3 -m pip install pipenv 
    pipenv --python 3.10 install
    python3 -m pip install datalad-installer
    datalad-installer --sudo ok git-annex -m datalad/git-annex:release
}

function datasets() {
    source ./bin/_datasets.sh "$@"
}

function gcp() {
    GCPUSAGE='Runs tasks at Google Cloud Platform.
    Provide task after gcp like this: run.sh gcp my-task [parameter ...].
    If no tasks provided (run.sh gcp) it syncs remote and local files (gets results).'
    if [[ -z ${AISCIBB_GCP_FLAG-} ]] ; then
        echo $GCPUSAGE
        # In local environment.
        [[ ! -z ${AISCIBB_GIT_TOKEN-} ]]  || ( echo "Please set AISCIBB_GIT_TOKEN environment variable (see https://github.com/settings/tokens)." ; exit 1 )
        [[ ! -z ${AISCIBB_GCP_SSH_USERHOST-} ]]  || ( echo "Please set AISCIBB_GCP_SSH_USERHOST environment variable (example: user@123.123.123.123)." ; exit 1 )

        echo Uploading files required to run commands...
        scp -q ~/.gitconfig scp://$AISCIBB_GCP_SSH_USERHOST/.gitconfig
        scp -q ./run.sh scp://$AISCIBB_GCP_SSH_USERHOST/run.sh
        if [[ $# -gt 0 ]]; then
            echo Calling GCP to run a task...
            ssh ssh://$AISCIBB_GCP_SSH_USERHOST 'AISCIBB_GCP_FLAG=1 bash' ./run.sh gcp $( git rev-parse --abbrev-ref HEAD ) "$AISCIBB_GIT_TOKEN" "$@"
        fi

        echo Getting results
        ssh ssh://$AISCIBB_GCP_SSH_USERHOST 'AISCIBB_GCP_FLAG=1 bash' ./run.sh gcp

        echo Syncing artifacts from GCP...
        # TODO: configure artifacts dir at remote.
        rsync --info=progress2 -r $AISCIBB_GCP_SSH_USERHOST:~/artifacts/\* $AISCBB_ARTIFACTS_DIR
        echo Done. Artifacts at $AISCBB_ARTIFACTS_DIR
    else
        # In remote environment.
        # If all prerequisits met:
        mkdir -vp $AISCBB_ARTIFACTS_DIR
        mkdir -vp $AISCBB_DATA_DIR

        C_ID_FILE=./aiscbb_container_id.cid
        if [[ $# -eq 0 ]]; then
            echo "Containers (tasks):"
            sudo docker ps
            if [[ -e $C_ID_FILE ]]; then
                CID=$( cat $C_ID_FILE ) 
                if sudo docker ps | grep $CID ; then 
                    C_LOG_FILE="$AISCBB_ARTIFACTS_DIR/$( date +%Y-%m-%d-%H%M )_run_sh.log "
                    sudo docker logs -f $CID 2>&1 | tee $C_LOG_FILE
                fi
            fi
        else
            echo [GCP] Run a task...
            echo [GCP] Stoping current task if any.
            if [[ -e $C_ID_FILE ]] ; then 
                CID=$( cat $C_ID_FILE ) 
                if sudo docker ps | grep $CID ; then 
                    echo There is $CID container running. Stoppping...
                    C_LOG_FILE="$AISCBB_ARTIFACTS_DIR/$( date +%Y-%m-%d-%H%M )_run_sh.log "
                    sudo docker logs -f $CID &> $C_LOG_FILE
                    sudo docker rm $CID
                fi
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
            ( cd $TARGETDIR ; sudo docker buildx build -t aiscbb . )

            echo Running container...
            shift 2  # Remove first two params for gcp.

            # See https://github.com/pytorch/xla/blob/master/docs/pjrt.md#docker
            #sudo docker container run \
            #    --privileged --net=host \
            sudo docker container run \
                -e AISCBB_ARTIFACTS_DIR=/aiscbb_artifacts \
                -e AISCBB_DATA_DIR=/asicbb_data \
                -v $AISCBB_ARTIFACTS_DIR:/aiscbb_artifacts \
                -v $AISCBB_DATA_DIR:/asicbb_data \
                -v ~/.gitconfig:/etc/gitconfig \
                --cidfile="$C_ID_FILE" \
                --detach \
                aiscbb bash run.sh "$@"
            CID=$( cat $C_ID_FILE )
            C_LOG_FILE="$AISCBB_ARTIFACTS_DIR/$( date +%Y-%m-%d-%H%M )_run_sh.log "
            sudo docker logs -f $CID 2>&1 | tee $C_LOG_FILE

            [[ ! -e %TARGETDIR ]] || rm -dr $TARGETDIR
            echo At GCP. Finished.
        fi
    fi
}

function train() {
    pipenv run python3 "$root_dir/src/main.py" "$@"
}

################################################################################


if [[ $# == 0 ]] ; then 
    echo "$USAGE"
else 
    cmd=$1
    shift 1
    $cmd "$@"  
fi

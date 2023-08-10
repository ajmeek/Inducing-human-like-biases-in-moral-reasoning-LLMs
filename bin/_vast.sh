
if [[ "${VAST_CONTAINERLABEL:-}" != "" ]] ; then
    _mamba
    mamba env create --force -n $PYTHON_ENV_NAME -f environment.yml -f environment-cuda.yml
    apt install netbase  # To enable /etc/protocols which is required by git-annex.
    pip install vastai
    mamba activate $PYTHON_ENV_NAME
    echo Done
else 
    [[ $# -eq 0 ]] && ( echo 'Usage: run.sh vast <GIT_TOKEN> <SSH_CONNECT_COMMAND>'  ; exit 1 )
    GIT_TOKEN=$1
    shift 1
    SSH_CMD=$*
    GITBRANCH=${CURRENT_GIT_BRANCH-$GIT_MAIN_BRANCH_NAME}
    CLONEDIR=/workspace/brainbias 
    
    echo 'Running at vast...'
    $SSH_CMD '/bin/bash -s' << EOF
    git config --global credential.helper 'cache --timeout=172800' # 48 hours
    git config --global user.name $( git config --global user.name )
    git config --global user.email $( git config --global user.email )
    echo 'Clonning...'
    git clone -b $GITBRANCH https://$GIT_TOKEN@$GIT_REMOTE $CLONEDIR
EOF
    echo 'Done clonning. Next loging and run `run.sh vast` to provision'
fi
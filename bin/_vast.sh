
CLONEDIR=/workspace/brainbias 
if [[ "${VAST_CONTAINERLABEL:-}" != "" ]] ; then
    _mamba

    echo 'Installing tools...'
    apt install -q nnn htop neovim 

    echo "Creating Python environment..."
    mamba env create --force -n $PYTHON_ENV_NAME -f environment-cuda.yml
    apt install -q netbase  # To enable /etc/protocols which is required by git-annex.

    echo "Activating Python environment..."
    mamba activate $PYTHON_ENV_NAME
    pip install -q vastai
    echo "Done"
else 
    [[ $# -le 1 || ! -v GITHUB_TOKEN ]] && (
        echo 'Usage: run.sh vast <SSH_CONNECT_COMMAND>
Environment variables:

    - GITHUB_TOKEN - GitHub token, https://github.com/settings/tokens
    - (Optional) WANDB_API_KEY - wandb.ai API key, https://docs.wandb.ai/guides/track/environment-variables#docusaurus_skipToContent_fallback
    - (Optional) HUGGING_FACE_HUB_TOKEN - https://huggingface.co/docs/hub/security-tokens
        '  
        exit 1 
    )
    SSH_CMD=$*
    GITBRANCH=${CURRENT_GIT_BRANCH-$GIT_MAIN_BRANCH_NAME}
    
    echo 'Running at vast...'
    $SSH_CMD '/bin/bash' -s << EOF

    export WANDB_API_KEY=${WANDB_API_KEY:-}
    export HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}
    echo >> ~/.bashrc
    echo "export WANDB_API_KEY=${WANDB_API_KEY:-}" >> ~/.bashrc
    echo "export HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" >> ~/.bashrc
    git config --global credential.helper 'cache --timeout=172800' # 48 hours
    git config --global user.name $( git config --global user.name )
    git config --global user.email $( git config --global user.email )

    echo 'Clonning...'
    git clone -b $GITBRANCH https://$GITHUB_TOKEN@$GIT_REMOTE $CLONEDIR
    echo 'Clonned to $CLONEDIR. Next: enter the remote shell and run "./run.sh vast"'
EOF
    echo 'Done'
fi
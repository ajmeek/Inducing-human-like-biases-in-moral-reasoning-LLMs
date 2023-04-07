[[ -e "$root_dir" ]] || ( echo 'root dir not found'  ; exit 1 )
datadir=$root_dir/data
[[ -e "$datadir" ]] || ( echo 'data dir not found'  ; exit 1 )

if [[ ! -e data/ds000212  || ! -e data/functional_flattened ]]; then 
    echo Downloading and processing ds000212...
    pushd "$datadir"
    datalad install --get-data https://github.com/OpenNeuroDatasets/ds000212.git
    popd
    python ./bin/fMRI_utils.py
    echo 'done'
fi

if [[ ! -e data/ethics ]]; then 
    echo Downloading and processing ETHICS...
    # Download and extract:
    pushd "$datadir"
    [[ -e ethics.tar ]] || curl -O 'https://people.eecs.berkeley.edu/~hendrycks/ethics.tar' 
    mkdir -p ethics
    tar -xvf ethics.tar 
    rm ./ethics.tar
    popd

    [[ -e "$root_dir/data/ethics/commonsense/cm_train.csv" ]] || ( echo 'downloading ethics ds failed'  ; exit 1 )
    echo 'done'
fi

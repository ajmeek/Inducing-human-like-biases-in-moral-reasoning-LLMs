[[ -e "$root_dir" ]] || ( echo "root dir not found: '$root_dir'"  ; exit 1 )
DATADIR=${AISCBB_DATA_DIR:-$root_dir/data}
[[ -e "$DATADIR" ]] || ( echo "data dir not found: '$DATADIR'"  ; exit 1 )

if [[ ! -e $DATADIR/ethics ]]; then 
    echo Downloading and processing ETHICS...
    pushd "$DATADIR"
    [[ -e ethics.tar ]] || curl -O 'https://people.eecs.berkeley.edu/~hendrycks/ethics.tar' 
    mkdir -p ethics
    tar -xvf ethics.tar 
    rm ./ethics.tar
    popd
    [[ -e "$DATADIR/ethics/commonsense/cm_train.csv" ]] || ( echo 'downloading ethics ds failed'  ; exit 1 )
    echo 'done'
fi

if [[ ! -e $DATADIR/ds000212  || ! -e $DATADIR/functional_flattened ]]; then 
    echo Downloading and processing ds000212...
    ( pushd "$DATADIR" ; datalad install --get-data https://github.com/OpenNeuroDatasets/ds000212.git ; popd )

    if [[ ! -e $DATADIR/scenarios.csv ]]; then
        cp $root_dir/data/scenarios.csv $DATADIR/scenarios.csv
    fi

    python3.9 ./bin/fMRI_utils.py
    echo 'done'
fi
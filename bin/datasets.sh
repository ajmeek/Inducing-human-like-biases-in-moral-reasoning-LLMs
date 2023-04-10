[[ -e "$root_dir" ]] || ( echo "root dir not found: '$root_dir'"  ; exit 1 )
datadir=$root_dir/data
pwd
ls 
echo $datadir
[[ -e "$datadir" ]] || ( echo "data dir not found: '$datadir'"  ; exit 1 )

exit

if [[ ! -e data/ethics ]]; then 
    echo Downloading and processing ETHICS...
    pushd "$datadir"
    [[ -e ethics.tar ]] || curl -O 'https://people.eecs.berkeley.edu/~hendrycks/ethics.tar' 
    mkdir -p ethics
    tar -xvf ethics.tar 
    rm ./ethics.tar
    popd
    [[ -e "$root_dir/data/ethics/commonsense/cm_train.csv" ]] || ( echo 'downloading ethics ds failed'  ; exit 1 )
    echo 'done'
fi

if [[ ! -e data/ds000212  || ! -e data/functional_flattened ]]; then 
    echo Downloading and processing ds000212...
    ( pushd "$datadir" ; datalad install --get-data https://github.com/OpenNeuroDatasets/ds000212.git ; popd )
    python ./bin/fMRI_utils.py
    echo 'done'
fi
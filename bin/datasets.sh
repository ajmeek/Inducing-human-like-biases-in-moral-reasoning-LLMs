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


if [[ ! -e $DATADIR/ds000212_scenarios.csv ]]; then
    cp $root_dir/data/ds000212_scenarios.csv $DATADIR/ds000212_scenarios.csv
fi

if [[ ! -e $DATADIR/ds000212 ]] ; then
    echo Downloading ds000212...
    ( pushd "$DATADIR" ; datalad install --get-data https://github.com/OpenNeuroDatasets/ds000212.git ; popd )
fi

num_cpus=$( python -c "import psutil; print(max(1, psutil.cpu_count(logical=True) - 2))" )
(
    export TARGET_DS_NAME=ds000212_raw
    export SOURCE_DS_NAME=ds000212
    export IS_ROI_ARG=--no-roi
    echo Running make for $TARGET_DS_NAME
    echo Shell in datasets.sh: $SHELL
    make -f ./bin/ds000212.mk all  --jobs $num_cpus --silent
)
(
    exit 0  # TODO
    export TARGET_DS_NAME=ds000212_roi
    export SOURCE_DS_NAME=ds000212
    export IS_ROI_ARG=--roi
    echo Running make for $TARGET_DS_NAME
    make -f ./bin/ds000212.mk all  --jobs $num_cpus  --silent
)

echo 'done'

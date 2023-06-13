
[[ -e "$root_dir" ]] || ( echo "root dir not found: '$root_dir'"  ; exit 1 )
[[ -e "$AISCBB_DATA_DIR" ]] || ( echo "data dir not found: '$AISCBB_DATA_DIR'"  ; exit 1 )

num_cpus=$( python3 -c "import psutil; print(max(1, psutil.cpu_count(logical=True) - 2))" )

function ethics() { 
if [[ ! -e $AISCBB_DATA_DIR/ethics ]]; then 
    echo Downloading and processing ETHICS...
    pushd "$AISCBB_DATA_DIR"
    [[ -e ethics.tar ]] || curl -O 'https://people.eecs.berkeley.edu/~hendrycks/ethics.tar' 
    mkdir -p ethics
    tar -xvf ethics.tar 
    rm ./ethics.tar
    popd
    [[ -e "$AISCBB_DATA_DIR/ethics/commonsense/cm_train.csv" ]] || ( echo 'downloading ethics ds failed'  ; exit 1 )
    echo 'done'
fi
}

function ds000212_scenarios() {
    if [[ ! -e $AISCBB_DATA_DIR/ds000212_scenarios.csv ]]; then
        cp $root_dir/data/ds000212_scenarios.csv $AISCBB_DATA_DIR/ds000212_scenarios.csv
    fi
}

function _ds000212_download() {
    if [[ ! -e $AISCBB_DATA_DIR/ds000212 ]] ; then
        echo Downloading ds000212...
        pipenv run datalad install -s https://github.com/OpenNeuroDatasets/ds000212.git $AISCBB_DATA_DIR/ds000212
    fi
}

function ds000212_raw() {
    _ds000212_download
    export TARGET_DS_NAME=ds000212_raw
    export SOURCE_DS_NAME=ds000212
    export IS_ROI_ARG=--no-roi
    echo Running make for $TARGET_DS_NAME
    make -f ./bin/ds000212.mk all  --jobs $num_cpus --silent
}

function ds000212_roi() {
    _ds000212_download
    export TARGET_DS_NAME=ds000212_roi
    export SOURCE_DS_NAME=ds000212
    export IS_ROI_ARG=--roi
    echo Running make for $TARGET_DS_NAME
    make -f ./bin/ds000212.mk all  --jobs $num_cpus  --silent
}

function ds000212_fmriprep() {
    _ds000212_download
    export TARGET_DS_NAME=ds000212_fmriprep
    export SOURCE_DS_NAME=ds000212/derivatives/ds000212-fmriprep
    if [[ ! -e $AISCBB_DATA_DIR/$SOURCE_DS_NAME ]] ; then
        echo Downloading $AISCBB_DATA_DIR/$SOURCE_DS_NAME...
        datalad install -s https://github.com/OpenNeuroDerivatives/ds000212-fmriprep.git "$AISCBB_DATA_DIR/$SOURCE_DS_NAME"
    fi
    echo Running make for $TARGET_DS_NAME
    make -f ./bin/ds000212prep.mk all  --jobs 1 
}

ethics
ds000212_scenarios

while [[ $# != 0 ]] ; do
    cmd=$1
    $cmd
    shift 1
done

echo 'done'

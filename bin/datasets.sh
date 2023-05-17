#!/bin/bash

[[ -e "$root_dir" ]] || ( echo "root dir not found: '$root_dir'"  ; exit 1 )
DATADIR=${AISCBB_DATA_DIR:-$root_dir/data}
[[ -e "$DATADIR" ]] || ( echo "data dir not found: '$DATADIR'"  ; exit 1 )

num_cpus=$( python -c "import psutil; print(max(1, psutil.cpu_count(logical=True) - 2))" )

function ethics() { 
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
}

function ds000212() {
    if [[ ! -e $DATADIR/ds000212_scenarios.csv ]]; then
        cp $root_dir/data/ds000212_scenarios.csv $DATADIR/ds000212_scenarios.csv
    fi

    if [[ ! -e $DATADIR/ds000212 ]] ; then
        echo Downloading ds000212...
        ( pushd "$DATADIR" ; datalad install --get-data https://github.com/OpenNeuroDatasets/ds000212.git ; popd )
    fi

    # (
    #     export TARGET_DS_NAME=ds000212_raw
    #     export SOURCE_DS_NAME=ds000212
    #     export IS_ROI_ARG=--no-roi
    #     echo Running make for $TARGET_DS_NAME
    #     make -f ./bin/ds000212.mk all  --jobs $num_cpus --silent
    # )
    # (
    #     export TARGET_DS_NAME=ds000212_roi
    #     export SOURCE_DS_NAME=ds000212
    #     export IS_ROI_ARG=--roi
    #     echo Running make for $TARGET_DS_NAME
    #     make -f ./bin/ds000212.mk all  --jobs $num_cpus  --silent
    # )

    if [[ ! -e $DATADIR/ds000212-fmriprep ]] ; then
        echo Downloading ds000212-fmriprep...
        pushd "$DATADIR" 
        datalad install https://github.com/OpenNeuroDerivatives/ds000212-fmriprep.git
        popd
    fi
    (
        export TARGET_DS_NAME=ds000212_fmriprep_done
        export SOURCE_DS_NAME=ds000212_fmriprep
        echo Running make for $TARGET_DS_NAME
        make -f ./bin/ds000212prep.mk all  --jobs $num_cpus --silent
    )
}


ethics
ds000212

echo 'done'
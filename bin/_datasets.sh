
[[ -e "$root_dir" ]] || ( echo "root dir not found: '$root_dir'"  ; exit 1 )
[[ -e "$AISCBB_DATADIR" ]] || ( echo "data dir not found: '$AISCBB_DATADIR'"  ; exit 1 )

num_cpus=$( pycmd -c "import psutil; print(max(1, psutil.cpu_count(logical=True) - 2))" )

function ethics() { 
if [[ ! -e $AISCBB_DATADIR/ethics ]]; then 
    echo Downloading and processing ETHICS...
    pushd "$AISCBB_DATADIR"
    [[ -e ethics.tar ]] || curl -O 'https://people.eecs.berkeley.edu/~hendrycks/ethics.tar' 
    mkdir -p ethics
    tar -xvf ethics.tar 
    rm ./ethics.tar
    popd
    [[ -e "$AISCBB_DATADIR/ethics/commonsense/cm_train.csv" ]] || ( echo 'downloading ethics ds failed'  ; exit 1 )
    echo 'done'
fi
}

function ds000212() {
    if [[ ! -e $AISCBB_DATADIR/ds000212_scenarios.csv ]]; then
        cp $root_dir/data/ds000212_scenarios.csv $AISCBB_DATADIR/ds000212_scenarios.csv
    fi

    if [[ ! -e $AISCBB_DATADIR/ds000212 ]] ; then
        echo Downloading ds000212...
        # TODO : make similar to 
        pecmd datalad install --get-data -s https://github.com/OpenNeuroDatasets/ds000212.git $AISCBB_DATADIR/ds000212
    fi

    (
        export TARGET_DS_NAME=ds000212_raw
        export SOURCE_DS_NAME=ds000212
        export IS_ROI_ARG=--no-roi
        echo Running make for $TARGET_DS_NAME
        make -f ./bin/ds000212.mk all  --jobs $num_cpus --silent
    )
    # (
    #     export TARGET_DS_NAME=ds000212_roi
    #     export SOURCE_DS_NAME=ds000212
    #     export IS_ROI_ARG=--roi
    #     echo Running make for $TARGET_DS_NAME
    #     make -f ./bin/ds000212.mk all  --jobs $num_cpus  --silent
    # )

    #(
    #    export TARGET_DS_NAME=ds000212_fmriprep
    #    export SOURCE_DS_NAME=ds000212/derivatives/ds000212-fmriprep
    #    if [[ ! -e $AISCBB_DATADIR/$SOURCE_DS_NAME ]] ; then
    #        echo Downloading $AISCBB_DATADIR/$SOURCE_DS_NAME...
    #        datalad install -s https://github.com/OpenNeuroDerivatives/ds000212-fmriprep.git "$AISCBB_DATADIR/$SOURCE_DS_NAME"
    #    fi
    #    echo Running make for $TARGET_DS_NAME
    #    #make -f ./bin/ds000212prep.mk all  --jobs $num_cpus --silent
    #    make -f ./bin/ds000212prep.mk all  --jobs 1 
    #)
}


ethics
ds000212

echo 'done'

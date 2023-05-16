# Makefile for the dataset in data/ds000212 that converts .nii.gz files into .npz files (Numpy files).
# Note: Intended to be run in a Linux environment. It is not guaranteed to work on other operating systems.

.ONESHELL:

niigz_script = bin/ds000212_process_niigz.py
target_name = $(TARGET_DS_NAME)
source_name = $(SOURCE_DS_NAME)
target_dir = $(AISCBB_DATA_DIR)/$(target_name)
source_dir = $(AISCBB_DATA_DIR)/$(source_name)
source_files = $(wildcard $(source_dir)/sub-*/func/*.nii.gz)
# Use existent files to get names for npz files 
# because just using wildcard '*' wont' resolve into anything.
target_files = $(patsubst $(source_dir)%.nii.gz,$(target_dir)%.npz,$(source_files))
# Normilized targets to signal that a .npz file was normilized.
normalized_targets = $(subst .npz,-normalized,$(target_files))
procesed_file = $(target_dir)/processed
# Default task. Finish when all .npz files are normilized.

all : $(normalized_targets)
	echo $$SHELL
	echo Done normalization.

$(normalized_targets) : $(procesed_file) 

# Task to run when all target .npz files were converted from .nii.gz.
# The task run per each target file is defined in a pattern rule.
$(procesed_file) : $(target_files)
	echo Done conversion from .nii.gz to .npz. 
    # This gets max value for the second dim among all .npz files:
	cat $(target_dir)/sub*/func/*-description.txt \
	 | sed -En '/data shape/s_.*\([0-9]+, *([0-9]+)\)_\1_p'  \
	 | sort -n -r \
	 | head -n 1 \
	 > $(procesed_file)

# Static pattern rule to run per each .nii.gz file to get .npz file.
$(target_dir)%.npz : $(source_dir)%.nii.gz
	from_niigz=$<
	if [[ ! "$$from_niigz" =~ "task-dis" ]]; then 
		# Take only dis (ignore false believe) 
		exit 0
	fi
	from_tsv_file=$(subst _bold.nii.gz,_events.tsv,$<)
	to_npz=$@
	mkdir -p $$( dirname $$to_npz )
	if [[ ! -e $$to_npz ]] ; then
		d_file=$(subst .npz,-description.txt,$@)
		python $(niigz_script) \
			"$$from_niigz" \
			"$$from_tsv_file" \
			"$$to_npz" \
			"$$d_file" \
			$(IS_ROI_ARG)
	fi

# Static pattern rule to run per each .npz file to get it normalized.
$(target_dir)%-normalized : $(target_dir)%.npz 
	max_len=$$( cat $(procesed_file) )
	python bin/normalize.py $$max_len "$<" "$@"
	
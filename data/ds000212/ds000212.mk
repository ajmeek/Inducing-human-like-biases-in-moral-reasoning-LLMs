# Makefile for the dataset in data/ds000212 that converts .nii.gz files into .npz files (Numpy files).
# Note: Intended to be run in a Linux environment. It is not guaranteed to work on other operating systems.

.ONESHELL:

niigz_script = ./ds000212_process_niigz.py
target_dir = $(TARGET_DS_NAME)
source_dir = $(SOURCE_DS_NAME)

# Only for 'dis' files:
source_files = $(wildcard $(source_dir)/sub-*/func/*dis_run*.nii.gz)

# Use existent files to get names for npz files 
# because just using wildcard '*' wont' resolve into anything.
target_files = $(patsubst $(source_dir)%.nii.gz,$(target_dir)%.npz,$(source_files))

# Normilized targets to signal that a .npz file was normilized.
normalized_targets = $(subst .npz,-normalized,$(target_files))
procesed_file = $(target_dir)/processed
tar_files = $((subst .npz,-normalized,$(target_files))

# Default task. Finish when all .npz files are normilized.
all : $(normalized_targets) 
	for f in $(source_dir)/sub-*/func/*tsv ; do cp "$$f" "$(target_dir)$${f#ds000212}" ; done
	for f in $(target_dir)/sub-*/func/*npz ; do rm -f $${f%_*}.tar ;  tar --remove-files --append -f $${f%_*}.tar $${f%_*}* ;  done

$(normalized_targets) : $(procesed_file) 

# Task to run when all target .npz files were converted from .nii.gz.
# The task run per each target file is defined in a pattern rule.
$(procesed_file) : $(target_files)
	# This gets max value for the second dim among all .npz files:
	cat $(target_dir)/sub*/func/*-description.txt \
	 | sed -En '/data shape/s_.*\([0-9]+, *([0-9]+)\)_\1_p'  \
	 | sort -n -r \
	 | head -n 1 \
	 > $(procesed_file)

# Static pattern rule to run per each .nii.gz file to get .npz file.
$(target_files) : $(target_dir)%.npz : $(source_dir)%.nii.gz
	from_niigz=$<
	from_tsv_file=$(subst _bold.nii.gz,_events.tsv,$<)
	to_npz=$@
	mkdir -p $$( dirname $$to_npz )
	d_file=$(subst .npz,-description.txt,$@)
	python3 $(niigz_script) \
		"$$from_niigz" \
		"$$from_tsv_file" \
		"$$to_npz" \
		"$$d_file" \
		$(IS_ROI_ARG)

# Static pattern rule to run per each .npz file to get it normalized.
$(normalized_targets) : $(target_dir)%-normalized : $(target_dir)%.npz 
	max_len=$$( cat $(procesed_file) )
	python3 ./normalize.py $$max_len "$<" "$@"
	
 $(source_files) :
	datalad get "$@"

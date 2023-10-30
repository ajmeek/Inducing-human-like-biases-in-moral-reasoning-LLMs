
.ONESHELL:

niigz_script = ./ds000212prep_process_niigz.py
target_dir = $(TARGET_DS_NAME)
source_dir = $(SOURCE_DS_NAME)
brain_source_niigz_files = $(wildcard $(source_dir)/sub-*/func/sub*dis_run*_res-2_desc-preproc_bold.nii.gz)
target_files = $(patsubst $(source_dir)%.nii.gz,$(target_dir)%.npz,$(brain_source_niigz_files))

all : $(target_files)
	echo $(target_dir)
	echo $(source_dir)
	echo brain_source_niigz_files: $(brain_source_niigz_files)
	echo target_files: $(target_files)


$(target_files) : $(target_dir)%.npz : $(source_dir)%.nii.gz
	brain_niigz=$<
	brain_json=$(subst .nii.gz,.json,$<)
	mask_niigz=$(subst preproc_bold,brain_mask,$<)
	mask_json=$(subst  .nii.gz,.json,$(mask_niigz))

	datalad get "$$brain_json"
	datalad get "$$mask_niigz"
	datalad get "$$mask_json"

	to_npz=$@
	mkdir -p "$$( dirname $$to_npz )"
	python3 $(niigz_script) "$$brain_niigz" "$$mask_niigz" "$$to_npz"
	
	datalad drop "$$brain_niigz"
	datalad drop "$$brain_json"
	datalad drop "$$mask_niigz"
	datalad drop "$$mask_json"
	echo Stoppiing
	exit 1

 $(source_dir)%.nii.gz :
	datalad get "$@"

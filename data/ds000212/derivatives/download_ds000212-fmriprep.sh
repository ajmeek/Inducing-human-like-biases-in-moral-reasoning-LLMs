#!/bin/bash


#List of files to be downloaded and their use:
#    sub-XX.html  - html reports generated for each subject detailing fmriprep outlines. If you want to look at the figures
#        in each subject, then run datalad get fmriprep/sub-XX/figures . Not putting that in here since all
#        figures for any one subject are 50-60 mb while the html is much smaller.
#    sub-XX_task_XX_run_XX_space-MNI152NLin2009cAsym_res_2_desc-brain_mask.json/.nii.gz
#        These are the brain masks calculated for the subject in Montreal Neurological Institute 152 space. Later on
#        we'll use these with nilearn to skull strip the BOLD signal for each subject and all their runs
#    sub-XX_task_XX_run_XX_space-MNI152NLin2009cAsym_res-2_desk-preproc_bold.json/.nii.gz
#        These are the actual bold signals for what we want! Once we perform skull stripping then this will be the base data to work off of

#download
#test on these for now. The whole script will cause downloads that will take up quite a bit of space
subjects=03 # 04 #05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 27 28 29 30 31 32 33 34 35 38 39 40 41 42 44 45 46 47)
runs=1 # 2 3 4 5 6)


#uncomment and run this block below when running for the first time.

#initial datalad query/download to get symlink files - you will need these before running the rest of the script
#datalad install https://github.com/OpenNeuroDerivatives/ds000212-fmriprep.git #commented out for testing
#echo "$(pwd)"
#echo "$(pwd)/ds000212-fmriprep"
#mv  "$(pwd)/ds000212-fmriprep" "$(pwd)/fmriprep" #so that this will work with BIDS standard latter. enforced naming conventions

#the current masks that I want alone with be about 2-3 gigs per subject. I'm testing this over a limited range because my laptop's hard drive is limited
for i in "${subjects[@]}"; do

  #individual subject html reports, without figures
  datalad get ./fmriprep/sub-${i}.html

  for j in "${runs[@]}"; do

    #first task

    #BOLD scans and header files
    echo "data for subject ${i}, run ${j}. Includes masks, header files, and the large, preprocessed BOLD scans"
    datalad get ./fmriprep/sub-${i}/func/sub-${i}_task-dis_run-${j}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz || echo "File does not exist for this subject"
    datalad get ./fmriprep/sub-${i}/func/sub-${i}_task-dis_run-${j}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.json || echo "File does not exist for this subject"

    #Binary brain masks for skull stripping, incl. header files
    datalad get ./fmriprep/sub-${i}/func/sub-${i}_task-dis_run-${j}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz || echo "File does not exist for this subject"
    datalad get ./fmriprep/sub-${i}/func/sub-${i}_task-dis_run-${j}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.json || echo "File does not exist for this subject"

    #second task, false belief - normally only 4 runs here, versus the six runs for the first task. expect some echos

    #BOLD scans and header files - for second task
    datalad get ./fmriprep/sub-${i}/func/sub-${i}_task-fb_run-${j}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz || echo "File does not exist for this subject"
    datalad get ./fmriprep/sub-${i}/func/sub-${i}_task-fb_run-${j}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.json || echo "File does not exist for this subject"

    #Binary brain masks for skull stripping, header files - for second task
    datalad get ./fmriprep/sub-${i}/func/sub-${i}_task-fb_run-${j}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz || echo "File does not exist for this subject"
    datalad get ./fmriprep/sub-${i}/func/sub-${i}_task-fb_run-${j}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.json || echo "File does not exist for this subject"

  done
done


#test datalad command
#datalad get ./fmriprep/sub-03/func/sub-03_task-dis_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
#echo "datalad get ./fmriprep/sub-04/func/sub-04_task-dis_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
#!/bin/bash
#
# Runs experiments on RoBERTa.
# https://huggingface.co/roberta-large

# Default experiment parameters:
DS2_TRAIN_SLICE='[:1000]'
ETHICS_EPOCHS=15
FMRI_EPOCHS=55
MODEL_PATH=bert-large-cased
WARM_UP=0.5
SAMPLING=LAST
ENAME=UNKNOWN

function experiment() {
    # Tunnel all through here.
    #
    WANDB_NAME="${MODEL_PATH}_${ENAME}_$(date +%y-%m-%d_%H%M)"
    export WANDB_NAME
    echo ===============================================================
    echo "$WANDB_NAME" 
    echo ===============================================================
    LAST_CKPT="artifacts/${WANDB_NAME}.ckpt"
    ./run.sh train \
        --accumulate_grad_batches 10 \
        --batch_size_all 8 \
        --check_val_every_n_epoch 2 \
        --checkpoint_path $CKPT_FILE \
        --ds1.name commonsense \
        --ds1.test.slicing '[:1000]' \
        --ds1.train.shuffle 1 \
        --ds1.train.slicing "$DS1_TRAIN_SLICE" \
        --ds1.validation.shuffle 0 \
        --ds1.validation.slicing '[:1000]' \
        --ds2.enable $ds2enable \
        --ds2.name "LFB-$SAMPLING" \
        --ds2.test.slicing '[:100%]' \
        --ds2.train.shuffle 1 \
        --ds2.train.slicing "$DS2_TRAIN_SLICE" \
        --find_bs 0 \
        --find_lr 0 \
        --last_checkpoint_path $LAST_CKPT \
        --lr 1e-5 \
        --lr_base_model_factor 1.0 \
        --lr_warm_up_steps $WARM_UP \
        --max_epochs $MAX_EPOCHS \
        --model_path "$MODEL_PATH" \
        --num_workers 0 \
        --profiler simple \
        --stepLR_gamma 0.99 \
        --strategy ddp_find_unused_parameters_true \
        --train_all $TRAIN_ALL \


}

#########################################
# No fine tuning

#ENAME="no_F_T"
#CKPT_FILE=
#DS1_TRAIN_SLICE="[:80%]"
#DS2_TRAIN_SLICE="[:80%]"
#MAX_EPOCHS=0
#TRAIN_ALL=0
#ds2enable=0
#experiment


#########################################
# fMRI and ETHICS 

ENAME="fmri_and_ethics_hm"
CKPT_FILE=
DS1_TRAIN_SLICE="[:80%]"
DS2_TRAIN_SLICE="[:80%]"
MAX_EPOCHS=7
TRAIN_ALL=1
ds2enable=1
OLD_WARM_UP=$WARM_UP
WARM_UP=1.0
experiment

exit 0

WARM_UP=$OLD_WARM_UP

#########################################
# Ethics-HM then fmri-HM

ENAME="Ethics-HM_then_fmri-HM"
CKPT_FILE=
DS1_TRAIN_SLICE="[:80%]"
MAX_EPOCHS=$ETHICS_EPOCHS
TRAIN_ALL=1
ds2enable=0
experiment

CKPT_FILE="$LAST_CKPT"
DS1_TRAIN_SLICE="[:0]"
MAX_EPOCHS=$FMRI_EPOCHS
TRAIN_ALL=1
ds2enable=1
experiment

#########################################
# fMRI then ETHICS

ENAME="fmri-hm_then_ethics-hm"
CKPT_FILE=
DS1_TRAIN_SLICE="[:0]"
DS2_TRAIN_SLICE="[:1000]"
MAX_EPOCHS=$FMRI_EPOCHS
TRAIN_ALL=1
ds2enable=1
experiment

CKPT_FILE="$LAST_CKPT"
DS1_TRAIN_SLICE="[:80%]"
DS2_TRAIN_SLICE="[:0]"
LAST_CKPT=
MAX_EPOCHS=$ETHICS_EPOCHS
TRAIN_ALL=1
ds2enable=0
experiment


#########################################
# Ethics-H then fmri-HM  -- TODO: this is not working because no head training happens
# export WANDB_NAME="$MODEL_PATH, Ethics-H then fmri-HM  $(date +%y-%m-%d\ %H:%M)"
# echo $WANDB_NAME
# 
# echo First a head on Ethics
# CKPT_FILE=
# DS1_TRAIN_SLICE="[:80%]"
# LAST_CKPT=$TRANSFER_CKPT
# MAX_EPOCHS=$ETHICS_EPOCHS
# FLR=1
# TRAIN_ALL=0
# ds2enable=0
# experiment
# 
# FLR=0
# LR=1e-5
# 
# echo Second, train on fMRI only, while testing on Ethics:
# export WANDB_NAME="$MODEL_PATH, Ethics-H then fmri-HM  $(date +%y-%m-%d\ %H:%M)"
# CKPT_FILE=$TRANSFER_CKPT
# DS1_TRAIN_SLICE="[:0]"
# LAST_CKPT=
# MAX_EPOCHS=$FMRI_EPOCHS
# SAMPLING=LAST
# TRAIN_ALL=1
# ds2enable=1
# experiment
# 

#########################################
# fmri-HM then Ethics-H

#echo First all on fMRI:
#CKPT_FILE=
#DS1_TRAIN_SLICE="[:0]"
#DS2_TRAIN_SLICE="[:1000]"
#LAST_CKPT=$TRANSFER_CKPT
#MAX_EPOCHS=$FMRI_EPOCHS
#SAMPLING=AVG
#TRAIN_ALL=1
#ds2enable=1
#experiment
#
#echo Second, train on ETHICS only:
#CKPT_FILE=$TRANSFER_CKPT
#DS1_TRAIN_SLICE="[:80%]"
#DS2_TRAIN_SLICE="[:0]"
#LAST_CKPT=
#MAX_EPOCHS=$ETHICS_EPOCHS
#TRAIN_ALL=1
#ds2enable=0
#experiment

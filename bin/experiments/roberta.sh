#!/bin/bash
#
# Runs experiments on RoBERTa.
# https://huggingface.co/roberta-large
ACC_GB=10
BS=8
DS2_TRAIN_SLICE='[:1000]'
ETHICS_EPOCHS=15
FBS=0
FLR=0
FMRI_EPOCHS=55
LR=1e-5
LRBMF=1.0
MODEL_PATH=roberta-large
WARM_UP=0.5

function experiment() {
    echo ===============================================================
    echo $MODEL_PATH $( date ) 
    echo ===============================================================
    ./run.sh train \
        --accumulate_grad_batches $ACC_GB \
        --batch_size_all $BS \
        --check_val_every_n_epoch 2 \
        --checkpoint_path $CKPT_FILE \
        --ds1.name commonsense \
        --ds1.test.slicing '[:1000]' \
        --ds1.train.shuffle 1 \
        --ds1.train.slicing $DS1_TRAIN_SLICE \
        --ds1.validation.shuffle 0 \
        --ds1.validation.slicing '[:1000]' \
        --ds2.enable $ds2enable \
        --ds2.name LFB-$SAMPLING \
        --ds2.test.slicing '[:100%]' \
        --ds2.train.shuffle 1 \
        --ds2.train.slicing $DS2_TRAIN_SLICE \
        --find_bs $FBS \
        --find_lr $FLR \
        --last_checkpoint_path $LAST_CKPT \
        --lr $LR \
        --lr_base_model_factor $LRBMF \
        --lr_warm_up_steps $WARM_UP \
        --max_epochs $MAX_EPOCHS \
        --model_path $MODEL_PATH \
        --num_workers 0 \
        --profiler simple \
        --stepLR_gamma 0.99 \
        --strategy ddp_find_unused_parameters_true \
        --train_all $TRAIN_ALL \


}


#########################################
# (fmri and Ethics)-HM

echo On fMRI and Ethics 
LAST_CKPT=artifacts/ethics-and-fmri.ckpt
CKPT_FILE=
DS1_TRAIN_SLICE="[:80%]"
DS2_TRAIN_SLICE="[:80%]"
LAST_CKPT=
MAX_EPOCHS=7
WARM_UP=1.0
TRAIN_ALL=1
ds2enable=0
experiment

exit 0

WARM_UP=0.5
TRANSFER_CKPT=artifacts/RoBERTa-Ethics-fmri.ckpt


#########################################
# fmri-HM then Ethics-HM

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

#echo Second, train on ETHICS only:
#CKPT_FILE=$TRANSFER_CKPT
#LAST_CKPT=artifacts/fmri_then_ethics.ckpt
#DS1_TRAIN_SLICE="[:80%]"
#DS2_TRAIN_SLICE="[:0]"
#LAST_CKPT=
#MAX_EPOCHS=20
#TRAIN_ALL=1
#ds2enable=0
#experiment


#########################################
# Ethics-H then fmri-HM

echo First a head on Ethics
CKPT_FILE=
DS1_TRAIN_SLICE="[:80%]"
LAST_CKPT=$TRANSFER_CKPT
MAX_EPOCHS=$ETHICS_EPOCHS
FLR=1
TRAIN_ALL=0
ds2enable=0
experiment

FLR=0
LR=1e-5

echo Second, train on fMRI only, while testing on Ethics:
CKPT_FILE=$TRANSFER_CKPT
DS1_TRAIN_SLICE="[:0]"
LAST_CKPT=
MAX_EPOCHS=$FMRI_EPOCHS
SAMPLING=LAST
TRAIN_ALL=1
ds2enable=1
experiment

echo  train on fMRI only, while testing on Ethics:
CKPT_FILE=$TRANSFER_CKPT
DS1_TRAIN_SLICE="[:0]"
LAST_CKPT=
MAX_EPOCHS=$FMRI_EPOCHS
SAMPLING=AVG
TRAIN_ALL=1
ds2enable=1
experiment

exit 0

#########################################
# Ethics-HM then fmri-HM

#echo First all on Ethics
#CKPT_FILE=
#DS1_TRAIN_SLICE="[:80%]"
#LAST_CKPT=$TRANSFER_CKPT
#MAX_EPOCHS=$ETHICS_EPOCHS
#TRAIN_ALL=1
#ds2enable=0
#experiment

echo Second, train on fMRI only, while testing on Ethics:
CKPT_FILE=./artifacts/RoBERTa_3hours_on_ethics_only.ckpt
DS1_TRAIN_SLICE="[:0]"
LAST_CKPT=
MAX_EPOCHS=$FMRI_EPOCHS
SAMPLING=LAST
TRAIN_ALL=1
ds2enable=1
experiment


#########################################
# fmri-HM then Ethics-H

echo First all on fMRI:
CKPT_FILE=
DS1_TRAIN_SLICE="[:0]"
DS2_TRAIN_SLICE="[:1000]"
LAST_CKPT=$TRANSFER_CKPT
MAX_EPOCHS=$FMRI_EPOCHS
SAMPLING=AVG
TRAIN_ALL=1
ds2enable=1
experiment

echo Second, train on ETHICS only:
CKPT_FILE=$TRANSFER_CKPT
DS1_TRAIN_SLICE="[:80%]"
DS2_TRAIN_SLICE="[:0]"
LAST_CKPT=
MAX_EPOCHS=$ETHICS_EPOCHS
TRAIN_ALL=1
ds2enable=0
experiment
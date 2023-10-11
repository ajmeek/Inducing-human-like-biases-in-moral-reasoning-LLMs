
function experiment() {
    echo ===============================================================
    echo $MODEL_PATH $( date ) 
    echo ===============================================================
    ./run.sh train \
        --num_workers 0 \
        --accumulate_grad_batches 1 \
        --batch_size_all $BS \
        --lr_warm_up_steps $WARM_UP \
        --check_val_every_n_epoch 1 \
        --checkpoint_path $CKPT_FILE \
        --ds1.name commonsense \
        --ds1.test.slicing '[:1000]' \
        --ds1.train.slicing $DS1_TRAIN_SLICE \
        --ds1.validation.shuffle 0 \
        --ds1.validation.slicing '[:1500]' \
        --ds2.enable $ds2enable \
        --ds2.name LFB-$SAMPLING \
        --ds2.test.slicing '[:50%]' \
        --ds2.train.shuffle 1 \
        --ds2.train.slicing '[:50%]' \
        --find_lr 0 \
        --find_bs 0 \
        --last_checkpoint_path $LAST_CKPT \
        --max_epochs $MAX_EPOCHS \
        --model_path $MODEL_PATH \
        --stepLR_gamma 0.99 \
        --train_all $TRAIN_ALL \
        --strategy auto \
        --profiler simple \
	--lr $LR \
    | tee ./artifacts/$( date +%Y%m%d-%H%M%S )

}


MODEL_PATH=bert-base-cased
LAST_CKPT=artifacts/train_head_on_ethics_$MODEL_PATH.ckpt
BS=32
ACC_GB=1
LR=2e-5
for SAMPLING in AVG LAST ; do 
for ds2enable in 0 1; do
    if [[ $ds2enable == 0 ]]; then
        TRAIN_ALL=1
        MAX_EPOCHS=10
        WARM_UP=0.3
        DS1_TRAIN_SLICE="[:50%]"
        CKPT_FILE=
    else
        CKPT_FILE=$LAST_CKPT
        LAST_CKPT=
        TRAIN_ALL=1
        MAX_EPOCHS=30
        WARM_UP=0.3
        DS1_TRAIN_SLICE="[:0]"
    fi
    experiment
done
done

exit 0

# https://github.com/google-research/bert
MODEL_PATH=bert-large-cased
LAST_CKPT=artifacts/train_head_on_ethics_$MODEL_PATH.ckpt
ACC_GB=1
LR=3e-4
BS=8
for SAMPLING in AVG LAST ; do 
for ds2enable in 0 1; do
    if [[ $ds2enable == 0 ]]; then
        TRAIN_ALL=1
        MAX_EPOCHS=10
        WARM_UP=0.3
        DS1_TRAIN_SLICE="[:50%]"
        CKPT_FILE=
    else
        CKPT_FILE=$LAST_CKPT
        LAST_CKPT=
        TRAIN_ALL=1
        MAX_EPOCHS=30
        WARM_UP=0.3
        DS1_TRAIN_SLICE="[:0]"
    fi
    experiment
done
done


# https://huggingface.co/microsoft/deberta-v3-large
MODEL_PATH=microsoft/deberta-v2-xlarge
LAST_CKPT=artifacts/train_head_on_ethics_DEBERTA.ckpt
ACC_GB=1
LR=6e-6
BS=8
for SAMPLING in AVG ; do 
for ds2enable in 0 1; do
    if [[ $ds2enable == 0 ]]; then
        TRAIN_ALL=1
        MAX_EPOCHS=10
        WARM_UP=0.3
        DS1_TRAIN_SLICE="[:50%]"
        CKPT_FILE=
    else
        CKPT_FILE=$LAST_CKPT
        LAST_CKPT=
        TRAIN_ALL=1
        MAX_EPOCHS=30
        WARM_UP=0.3
        DS1_TRAIN_SLICE="[:0]"
    fi
    experiment
done
done

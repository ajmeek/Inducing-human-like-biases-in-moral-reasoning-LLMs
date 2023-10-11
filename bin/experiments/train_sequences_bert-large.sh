LAST_CKPT=artifacts/train_head_on_ethics_bert_large.ckpt

for ds2enable in 0 1; do

    if [[ $ds2enable == 0 ]]; then
        echo Second dataset disabled.
        TRAIN_ALL=1
        MAX_EPOCHS=10
        WARM_UP=1.0
        DS1_TRAIN_SLICE="[:100%]"
	FIND_LR=1
	BS=23
	ACC_GB=5
	CKPT_FILE=
    else
        echo Second dataset enabled.
        CKPT_FILE=$LAST_CKPT
	LAST_CKPT=
        TRAIN_ALL=1
        MAX_EPOCHS=100
        WARM_UP=0.75
        DS1_TRAIN_SLICE="[:0]"
	FIND_LR=1
	BS=23
	ACC_GB=5
    fi

    ./run.sh train \
	--num_workers 0 \
    	--torch_float32_matmul_precision high \
        --accumulate_grad_batches $ACC_GB \
        --batch_size_all $BS \
        --lr_warm_up_steps $WARM_UP \
        --check_val_every_n_epoch 3 \
        --checkpoint_path $CKPT_FILE \
        --ds1.name commonsense \
        --ds1.test.slicing '[:2000]' \
        --ds1.train.slicing $DS1_TRAIN_SLICE \
        --ds1.validation.shuffle 0 \
        --ds1.validation.slicing '[:3000]' \
        --ds2.enable $ds2enable \
        --ds2.name LFB-AVG \
        --ds2.test.slicing '[:100%]' \
        --ds2.train.shuffle 1 \
        --ds2.train.slicing '[:100%]' \
        --find_lr $FIND_LR \
	--lr 0.006918309709189364 \
        --last_checkpoint_path $LAST_CKPT \
        --max_epochs $MAX_EPOCHS \
        --model_path 'bert-large-cased' \
        --stepLR_gamma 0.99 \
        --stepLR_step_size 100 \
        --train_all $TRAIN_ALL \
        --strategy ddp_find_unused_parameters_true \

done

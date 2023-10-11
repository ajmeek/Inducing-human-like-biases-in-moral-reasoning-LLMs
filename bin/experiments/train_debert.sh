# See https://arxiv.org/pdf/2006.03654.pdf

for ds2enable in True False ; do

    ./run.sh train \
        --check_val_every_n_epoch 3 \
        --ds1.name commonsense \
        --ds1.test.batch_size 2 \
        --ds1.test.slicing '[-500:]' \
        --ds1.train.batch_size 2 \
        --ds1.train.slicing '[-1000:]' \
        --ds1.validation.batch_size 2 \
        --ds1.validation.slicing '[:1000]' \
        --ds2.train.batch_size 2 \
        --ds2.train.slicing '[:1000]' \
        --ds2.enable $ds2enable \
        --enable_checkpointing True \
        --before_lr_decay_warm_up_steps 5000 \
        --find_learning_rate \
        --train_all True \
        --lr 2e-4 \
        --model_path 'microsoft/deberta-v2-xlarge' \
        --sampling_method LAST \
        --accumulate_grad_batches 15 \
	--max_epochs 30 \
        --train_all True

done

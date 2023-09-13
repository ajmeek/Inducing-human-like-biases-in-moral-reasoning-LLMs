./run.sh train \
    --check_val_every_n_epoch 3 \
    --ds1.name commonsense \
    --ds1.test.slicing '[:0]' \
    --ds1.train.batch_size 5 \
    --ds1.train.slicing '[:200]' \
    --ds1.validation.batch_size 5 \
    --ds1.validation.slicing '[:200]' \
    --ds2.train.batch_size 5 \
    --lr 1e-4 \
    --max_epochs 100 \
    --model_path 'bert-large-cased' \
    --sampling_method LAST \
    --enable_checkpointing True \
    --before_lr_decay_warm_up_steps 500 \
    --train_all True

./run.sh train \
    --check_val_every_n_epoch 3 \
    --ds1.name commonsense \
    --ds1.test.slicing '[:0]' \
    --ds1.train.batch_size 5 \
    --ds1.train.slicing '[:200]' \
    --ds1.validation.batch_size 5 \
    --ds1.validation.slicing '[:200]' \
    --ds2.enable False \
    --lr 1e-4 \
    --max_epochs 100 \
    --model_path 'bert-large-cased' \
    --sampling_method LAST \
    --enable_checkpointing True \
    --before_lr_decay_warm_up_steps 500 \
    --train_all True



./run.sh train \
    --check_val_every_n_epoch 3 \
    --ds1.name commonsense \
    --ds1.test.slicing '[:0]' \
    --ds1.train.batch_size 5 \
    --ds1.train.slicing '[:2000]' \
    --ds1.validation.batch_size 5 \
    --ds1.validation.slicing '[:1000]' \
    --ds2.train.batch_size 5 \
    --has_ReduceLROnPlateau True \
    --lr 1e-4 \
    --lr_scheduler_frequency 10 \
    --lr_scheduler_interval epoch \
    --max_epochs 1000 \
    --model_path 'bert-large-cased' \
    --sampling_method SENTENCES \
    --enable_checkpointing True \
    --train_all True

./run.sh train \
    --check_val_every_n_epoch 3 \
    --ds1.name commonsense \
    --ds1.test.slicing '[:0]' \
    --ds1.train.batch_size 5 \
    --ds1.train.slicing '[:2000]' \
    --ds1.validation.batch_size 5 \
    --ds1.validation.slicing '[:1000]' \
    --ds2.enable False \
    --has_ReduceLROnPlateau True \
    --lr 1e-4 \
    --lr_scheduler_frequency 10 \
    --lr_scheduler_interval epoch \
    --max_epochs 1000 \
    --model_path 'bert-large-cased' \
    --sampling_method SENTENCES \
    --enable_checkpointing True \
    --train_all True


vastai stop instance $CONTAINER_ID;

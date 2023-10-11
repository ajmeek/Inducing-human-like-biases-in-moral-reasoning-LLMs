./run.sh train \
    --ds1.train.batch_size 10 \
    --ds2.train.batch_size 10 \
    --find_lr \
    --max_epochs -1 \
    --num_sanity_val_steps 2 \
    --overfit_batches 2 \
    --profiler simple \
    --train_all 1 \


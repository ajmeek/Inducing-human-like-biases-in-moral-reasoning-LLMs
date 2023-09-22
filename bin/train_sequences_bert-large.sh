

echo "First train only a head on Ethics"

./run.sh train \
    --check_val_every_n_epoch 3 \
    --ds1.name commonsense \
    --ds1.test.slicing '[:1000]' \
    --ds1.train.batch_size 6 \
    --ds1.train.slicing '[:6000]' \
    --ds1.validation.batch_size 5 \
    --ds1.validation.slicing '[:2000]' \
    --ds2.enable False \
    --max_epochs 20 \
    --model_path 'bert-large-cased' \
    --before_lr_decay_warm_up_steps 100 \
    --find_learning_rate 1 \
    --accumulate_grad_batches 10 \
    --stepLR_gamma 0.99 \
    --stepLR_step_size 10 \
    --train_all False \
    --last_checkpoint_path artifacts/train_head_on_ethics_bert_large.ckpt

echo "Now train all on fMRI only and see if Ethics accuracy improves"

./run.sh train \
    --check_val_every_n_epoch 3 \
    --ds1.name commonsense \
    --ds1.test.slicing '[:1000]' \
    --ds1.train.batch_size 6 \
    --ds1.train.slicing '[:0]' \
    --ds1.validation.batch_size 5 \
    --ds1.validation.slicing '[:2000]' \
    --ds2.enable True \
    --ds2.name LFB-AVG \
    --ds2.train.slicing '[:100%]' \
    --ds2.train.batch_size 4 \
    --ds2.train.shuffle 1 \
    --ds2.test.slicing '[:100%]' \
    --ds2.train.batch_size 4 \
    --max_epochs 100 \
    --model_path 'bert-large-cased' \
    --before_lr_decay_warm_up_steps 100 \
    --find_learning_rate 1 \
    --accumulate_grad_batches 100 \
    --stepLR_gamma 0.99 \
    --stepLR_step_size 10 \
    --train_all True \
    --checkpoint_path artifacts/train_head_on_ethics_bert_large.ckpt
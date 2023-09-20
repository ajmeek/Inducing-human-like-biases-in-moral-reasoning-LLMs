
echo Training using large batches as they did for RoBERTa

./run.sh train \
    --check_val_every_n_epoch 3 \
    --ds1.name commonsense \
    --ds1.test.slicing '[:1000]' \
    --ds1.train.batch_size 6 \
    --ds1.train.slicing '[:6000]' \
    --ds1.validation.batch_size 5 \
    --ds1.validation.slicing '[:2000]' \
    --ds2.name LFB-SENTENCES \
    --ds2.train.slicing '[:4000]' \
    --ds2.train.batch_size 4 \
    --ds2.train.shuffle 1 \
    --max_epochs 20 \
    --model_path 'bert-large-cased' \
    --enable_checkpointing True \
    --before_lr_decay_warm_up_steps 100 \
    --find_learning_rate 0 \
    --lr 1e-3 \
    --accumulate_grad_batches 100 \
    --stepLR_gamma 0.99 \
    --stepLR_step_size 10 \
    --profiler simple \
    --train_all True

    #--lr 0.00000120226443461741 \
exit 0

echo Make 'sweep' on sampling and accumulated gradients:

#for SM in LAST MIDDLE AVG SENTENCES ; do
#for BS in 2 4 8 ; do

./run.sh train \
    --check_val_every_n_epoch 3 \
    --ds1.name commonsense \
    --ds1.test.slicing '[:1000]' \
    --ds1.train.batch_size 5 \
    --ds1.train.slicing '[:1000]' \
    --ds1.validation.batch_size 5 \
    --ds1.validation.slicing '[:1000]' \
    --ds2.name LFB-AVG \
    --ds2.train.slicing '[:1000]' \
    --ds2.train.batch_size 4 \
    --ds2.train.shuffle 1 \
    --max_epochs 100 \
    --model_path 'bert-large-cased' \
    --enable_checkpointing True \
    --before_lr_decay_warm_up_steps 2000 \
    --lr 0.00000120226443461741 \
    --find_learning_rate 0 \
    --accumulate_grad_batches 8 \
    --stepLR_gamma 0.99 \
    --stepLR_step_size 100 \
    --train_all True

#done
#done

exit 0

echo Train only ETHICS:

./run.sh train \
    --check_val_every_n_epoch 3 \
    --ds1.name commonsense \
    --ds1.test.slicing '[:1000]' \
    --ds1.train.batch_size 5 \
    --ds1.train.slicing '[:1000]' \
    --ds1.validation.batch_size 5 \
    --ds1.validation.slicing '[:1000]' \
    --ds2.name LFB-LAST \
    --ds2.enable False \
    --ds2.train.slicing '[:1000]' \
    --ds2.train.batch_size 4 \
    --ds2.train.shuffle 1 \
    --lr 1e-4 \
    --max_epochs 25 \
    --model_path 'bert-large-cased' \
    --enable_checkpointing True \
    --before_lr_decay_warm_up_steps 5000 \
    --find_learning_rate \
    --accumulate_grad_batches 4 \
    --stepLR_gamma 0.99 \
    --stepLR_step_size 100 \
    --train_all True

exit 0

echo Train with DS2 test enabled:

./run.sh train \
    --check_val_every_n_epoch 3 \
    --ds1.name commonsense \
    --ds1.test.slicing '[:0]' \
    --ds1.train.batch_size 5 \
    --ds1.train.slicing '[:1000]' \
    --ds1.validation.batch_size 5 \
    --ds1.validation.slicing '[:1000]' \
    --ds2.name LFB-LAST \
    --ds2.train.slicing '[:1000]' \
    --ds2.train.batch_size 4 \
    --ds2.train.shuffle 1 \
    --ds2.test.slicing '[:200]' \
    --ds2.test.batch_size 2 \
    --ds2.test.shuffle 1 \
    --lr 1e-4 \
    --max_epochs 25 \
    --model_path 'bert-large-cased' \
    --enable_checkpointing True \
    --before_lr_decay_warm_up_steps 5000 \
    --find_learning_rate \
    --accumulate_grad_batches 8 \
    --stepLR_gamma 0.99 \
    --stepLR_step_size 100 \
    --train_all True




exit 0

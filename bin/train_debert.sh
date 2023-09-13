# See https://arxiv.org/pdf/2006.03654.pdf

./run.sh train \
    --check_val_every_n_epoch 3 \
    --ds1.name commonsense \
    --ds1.test.batch_size 2 \
    --ds1.test.slicing '[:500]' \
    --ds1.train.batch_size 1 \
    --ds1.train.slicing '[:1000]' \
    --ds1.validation.batch_size 2 \
    --ds1.validation.slicing '[:1000]' \
    --ds2.train.batch_size 1 \
    --enable_checkpointing True \
    --has_learning_rate_decay True \
    --limit_train_batches 700 \
    --limit_val_batches 500 \
    --lr 2e-4 \
    --eps 1e-5 \
    --max_epochs 400 \
    --model_path 'microsoft/deberta-v2-xlarge' \
    --sampling_method SENTENCES \
    --train_all True


./run.sh train \
    --check_val_every_n_epoch 3 \
    --ds1.name commonsense \
    --ds1.test.batch_size 2 \
    --ds1.test.slicing '[:500]' \
    --ds1.train.batch_size 1 \
    --ds1.train.slicing '[:1000]' \
    --ds1.validation.batch_size 2 \
    --ds1.validation.slicing '[:1000]' \
    --ds2.enable False \
    --enable_checkpointing True \
    --has_learning_rate_decay True \
    --limit_train_batches 700 \
    --limit_val_batches 500 \
    --lr 2e-4 \
    --eps 1e-5 \
    --max_epochs 400 \
    --model_path 'microsoft/deberta-v2-xlarge' \
    --sampling_method SENTENCES \
    --train_all True

./run.sh train \
    --check_val_every_n_epoch 3 \
    --ds1.name commonsense \
    --ds1.test.batch_size 2 \
    --ds1.test.slicing '[:500]' \
    --ds1.train.batch_size 1 \
    --ds1.train.slicing '[:1000]' \
    --ds1.validation.batch_size 2 \
    --ds1.validation.slicing '[:1000]' \
    --ds2.train.batch_size 1 \
    --enable_checkpointing True \
    --has_learning_rate_decay True \
    --limit_train_batches 700 \
    --limit_val_batches 500 \
    --lr 2e-4 \
    --eps 1e-5 \
    --max_epochs 400 \
    --model_path 'microsoft/deberta-v2-xlarge' \
    --sampling_method LAST \
    --train_all True


./run.sh train \
    --check_val_every_n_epoch 3 \
    --ds1.name commonsense \
    --ds1.test.batch_size 2 \
    --ds1.test.slicing '[:500]' \
    --ds1.train.batch_size 1 \
    --ds1.train.slicing '[:1000]' \
    --ds1.validation.batch_size 2 \
    --ds1.validation.slicing '[:1000]' \
    --ds2.enable False \
    --enable_checkpointing True \
    --has_learning_rate_decay True \
    --limit_train_batches 700 \
    --limit_val_batches 500 \
    --lr 2e-4 \
    --eps 1e-5 \
    --max_epochs 400 \
    --model_path 'microsoft/deberta-v2-xlarge' \
    --sampling_method LAST \
    --train_all True


./run.sh train \
    --check_val_every_n_epoch 3 \
    --ds1.name justice \
    --ds1.test.batch_size 2 \
    --ds1.test.slicing '[:500]' \
    --ds1.train.batch_size 1 \
    --ds1.train.slicing '[:1000]' \
    --ds1.validation.batch_size 2 \
    --ds1.validation.slicing '[:1000]' \
    --ds2.train.batch_size 1 \
    --enable_checkpointing True \
    --has_learning_rate_decay True \
    --limit_train_batches 700 \
    --limit_val_batches 500 \
    --lr 2e-4 \
    --eps 1e-5 \
    --max_epochs 400 \
    --model_path 'microsoft/deberta-v2-xlarge' \
    --sampling_method SENTENCES \
    --train_all True


./run.sh train \
    --check_val_every_n_epoch 3 \
    --ds1.name justice \
    --ds1.test.batch_size 2 \
    --ds1.test.slicing '[:500]' \
    --ds1.train.batch_size 1 \
    --ds1.train.slicing '[:1000]' \
    --ds1.validation.batch_size 2 \
    --ds1.validation.slicing '[:1000]' \
    --ds2.enable False \
    --enable_checkpointing True \
    --has_learning_rate_decay True \
    --limit_train_batches 700 \
    --limit_val_batches 500 \
    --lr 2e-4 \
    --eps 1e-5 \
    --max_epochs 400 \
    --model_path 'microsoft/deberta-v2-xlarge' \
    --sampling_method SENTENCES \
    --train_all True



./run.sh train \
    --check_val_every_n_epoch 3 \
    --ds1.name justice \
    --ds1.test.batch_size 2 \
    --ds1.test.slicing '[:500]' \
    --ds1.train.batch_size 1 \
    --ds1.train.slicing '[:1000]' \
    --ds1.validation.batch_size 2 \
    --ds1.validation.slicing '[:1000]' \
    --ds2.train.batch_size 1 \
    --enable_checkpointing True \
    --has_learning_rate_decay True \
    --limit_train_batches 700 \
    --limit_val_batches 500 \
    --lr 2e-4 \
    --eps 1e-5 \
    --max_epochs 400 \
    --model_path 'microsoft/deberta-v2-xlarge' \
    --sampling_method LAST \
    --train_all True


./run.sh train \
    --check_val_every_n_epoch 3 \
    --ds1.name justice \
    --ds1.test.batch_size 2 \
    --ds1.test.slicing '[:500]' \
    --ds1.train.batch_size 1 \
    --ds1.train.slicing '[:1000]' \
    --ds1.validation.batch_size 2 \
    --ds1.validation.slicing '[:1000]' \
    --ds2.enable False \
    --enable_checkpointing True \
    --has_ReduceLROnPlateau True \
    --limit_train_batches 700 \
    --limit_val_batches 500 \
    --lr 2e-4 \
    --lr_scheduler_frequency 10 \
    --lr_scheduler_interval epoch \
    --eps 1e-5 \
    --max_epochs 400 \
    --model_path 'microsoft/deberta-v2-xlarge' \
    --sampling_method LAST \
    --train_all True

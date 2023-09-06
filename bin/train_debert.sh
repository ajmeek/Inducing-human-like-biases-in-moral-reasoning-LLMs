./run.sh train --max_epochs 3000 --ds1.train.batch_size 1 --ds1.validation.batch_size 2 --ds1.test.batch_size 2 --ds1.train.slicing '[50%:]' --ds1.validation.slicing '[60%:]' --ds1.test.slicing '[:40%]' --train_all True --check_val_every_n_epoch 3 --ds2.train.batch_size 1 --model_path 'microsoft/deberta-v2-xlarge' --ds1.name commonsense --early_stop_threshold 0.8  --sampling_method SENTENCES --precision 16-mixed

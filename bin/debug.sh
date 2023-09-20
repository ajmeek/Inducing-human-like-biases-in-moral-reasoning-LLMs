./run.sh train --max_epochs -1 --num_sanity_val_steps 2 --lr 5e-4 --ds1.train.batch_size 10 --ds2.train.batch_size 10 --train_all 1 --profiler simple --find_learning_rate  --overfit_batches 2

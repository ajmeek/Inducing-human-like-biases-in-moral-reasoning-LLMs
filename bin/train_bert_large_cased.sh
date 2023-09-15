for ds2enable in True False ; do
	./run.sh train \
	    --check_val_every_n_epoch 3 \
	    --ds1.name commonsense \
	    --ds1.test.slicing '[:100]' \
	    --ds1.train.batch_size 5 \
	    --ds1.train.slicing '[:1000]' \
	    --ds1.validation.batch_size 5 \
	    --ds1.validation.slicing '[:500]' \
	    --ds2.train.batch_size 5 \
	    --ds2.enable $ds2enable \
	    --lr 1e-4 \
	    --max_epochs 100 \
	    --model_path 'bert-large-cased' \
	    --sampling_method LAST \
	    --enable_checkpointing True \
	    --before_lr_decay_warm_up_steps 7500 \
	    --find_learning_rate \
	    --accumulate_grad_batches 7 \
	    --stepLR_gamma 0.99 \
	    --stepLR_step_size 100 \
	    --train_all True

done

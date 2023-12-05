# %%
import pandas as pd
import numpy as np
from pandas import json_normalize
import wandb
import json
from pprint import pp

api = wandb.Api()
# %%
# Project is specified by <entity/project-name>
runs = api.runs("asdfasdfasdfdsafsd/AISC_BB")

summary_list, config_list, name_list = [], [], []
data_list = []
for run in runs:
    if "_runtime" not in run.summary:
        continue
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    # summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    # config_list.append(
    #    {k: v for k,v in run.config.items()
    #      if not k.startswith('_')})

    # .name is the human-readable name of the run.
    # name_list.append(run.name)

    s = run.summary._json_dict
    c = {k: v for k, v in run.config.items() if not k.startswith("_")}
    data_list.append(s | c | {"name": run.name})

runs_df = pd.DataFrame(data_list)

# %%
# Sort columns by name:
runs_df = runs_df.reindex(sorted(runs_df.columns), axis=1)

# %%
# To csv:
runs_df.to_csv("project.csv")


# %%
"""
LFB-AVG-test-CosineSimilarity	LFB-AVG-test-CosineSimilarity/dataloader_idx_1	LFB-AVG-test-MeanAbsoluteError	LFB-AVG-test-MeanAbsoluteError/dataloader_idx_1	LFB-AVG-test-MeanSquaredError	LFB-AVG-test-MeanSquaredError/dataloader_idx_1	LFB-AVG-test-behavior-MulticlassAUROC	LFB-AVG-test-behavior-MulticlassAccuracy	LFB-AVG-test-behavior-MulticlassF1Score	LFB-AVG-test-label-CosineSimilarity	LFB-AVG-test-label-MeanAbsoluteError	LFB-AVG-test-label-MeanSquaredError	LFB-LAST-test-CosineSimilarity	LFB-LAST-test-CosineSimilarity/dataloader_idx_1	LFB-LAST-test-MeanAbsoluteError	LFB-LAST-test-MeanAbsoluteError/dataloader_idx_1	LFB-LAST-test-MeanSquaredError	LFB-LAST-test-MeanSquaredError/dataloader_idx_1	LFB-LAST-test-behavior-MulticlassAUROC	LFB-LAST-test-behavior-MulticlassAccuracy	LFB-LAST-test-behavior-MulticlassF1Score	LFB-LAST-test-label-CosineSimilarity	LFB-LAST-test-label-MeanAbsoluteError	LFB-LAST-test-label-MeanSquaredError	LFB-LAST-validation-MeanSquaredError	LFB-SENTENCES-test-behavior-MulticlassAUROC	LFB-SENTENCES-test-behavior-MulticlassAccuracy	LFB-SENTENCES-test-behavior-MulticlassF1Score	LFB-SENTENCES-test-label-CosineSimilarity	LFB-SENTENCES-test-label-MeanAbsoluteError	LFB-SENTENCES-test-label-MeanSquaredError	_runtime	_step	_timestamp	_wandb	artifactspath	batch_size	batches_per_epoch	bs_encoder.layer.10.output.dense	bs_encoder.layer.11.output.dense	bs_encoder.layer.12.output.dense	bs_encoder.layer.13.output.dense	bs_encoder.layer.14.output.dense	bs_encoder.layer.15.output.dense	bs_encoder.layer.16.output.dense	bs_encoder.layer.17.output.dense	bs_encoder.layer.18.output.dense	bs_encoder.layer.19.output.dense	bs_encoder.layer.2.output.dense	bs_encoder.layer.20.output.dense	bs_encoder.layer.21.output.dense	bs_encoder.layer.22.output.dense	bs_encoder.layer.23.output.dense	bs_encoder.layer.3.output.dense	bs_encoder.layer.4.output.dense	bs_encoder.layer.5.output.dense	bs_encoder.layer.6.output.dense	bs_encoder.layer.7.output.dense	bs_encoder.layer.8.output.dense	bs_encoder.layer.9.output.dense	bs_hidden_state_0	bs_hidden_state_1	bs_hidden_state_10	bs_hidden_state_11	bs_hidden_state_12	bs_hidden_state_2	bs_hidden_state_3	bs_hidden_state_4	bs_hidden_state_5	bs_hidden_state_6	bs_hidden_state_7	bs_hidden_state_8	bs_hidden_state_9	bs_hs_0	bs_hs_1	bs_hs_10	bs_hs_11	bs_hs_12	bs_hs_13	bs_hs_14	bs_hs_15	bs_hs_16	bs_hs_17	bs_hs_18	bs_hs_19	bs_hs_2	bs_hs_20	bs_hs_21	bs_hs_22	bs_hs_23	bs_hs_24	bs_hs_3	bs_hs_4	bs_hs_5	bs_hs_6	bs_hs_7	bs_hs_8	bs_hs_9	checkpoint	checkpoint_path	checkpointing	cod_hidden_state_0	cod_hidden_state_1	cod_hidden_state_10	cod_hidden_state_11	cod_hidden_state_12	cod_hidden_state_2	cod_hidden_state_3	cod_hidden_state_4	cod_hidden_state_5	cod_hidden_state_6	cod_hidden_state_7	cod_hidden_state_8	cod_hidden_state_9	cod_hs_0	cod_hs_1	cod_hs_10	cod_hs_11	cod_hs_12	cod_hs_13	cod_hs_14	cod_hs_15	cod_hs_16	cod_hs_17	cod_hs_18	cod_hs_19	cod_hs_2	cod_hs_20	cod_hs_21	cod_hs_22	cod_hs_23	cod_hs_24	cod_hs_3	cod_hs_4	cod_hs_5	cod_hs_6	cod_hs_7	cod_hs_8	cod_hs_9	commonsense-test-MulticlassAUROC	commonsense-test-MulticlassAUROC/dataloader_idx_0	commonsense-test-MulticlassAccuracy	commonsense-test-MulticlassAccuracy/dataloader_idx_0	commonsense-test-MulticlassF1Score	commonsense-test-label-MulticlassAUROC	commonsense-test-label-MulticlassAccuracy	commonsense-test-label-MulticlassF1Score	commonsense-validation-MulticlassAUROC	commonsense-validation-MulticlassAccuracy	commonsense-validation-MulticlassF1Score	commonsense-validation-label-MulticlassAUROC	commonsense-validation-label-MulticlassAccuracy	commonsense-validation-label-MulticlassF1Score	datapath	debug	ds1	ds1/enable	ds1/input_col	ds1/label_col	ds1/loss_fn	ds1/name	ds1/path	ds1/revision	ds1/test/batch_size	ds1/test/shuffle	ds1/test/slicing	ds1/train/batch_size	ds1/train/shuffle	ds1/train/slicing	ds1/validation/batch_size	ds1/validation/shuffle	ds1/validation/slicing	ds2	ds2/enable	ds2/input_col	ds2/label_col	ds2/loss_fn	ds2/name	ds2/path	ds2/revision	ds2/sampling_method	ds2/test	ds2/train/batch_size	ds2/train/shuffle	ds2/train/slicing	ds2/validation	early_stop_threshold	epoch	find_bs	find_learning_rate	find_lr	finetuned_path	last_checkpoint_path	loss_names	loss_weights	lr	lr-AdamW	lr-AdamW/pg1	lr-AdamW/pg2	lr-AdamW/pg3	lr-AdamW/pg4	model_path	name	num_epochs	num_samples_test	num_samples_train	num_workers	only_train_head	plc	plc/adamw/betas	plc/adamw/eps	plc/adamw/lr	plc/adamw/weight_decay	plc/before_lr_decay_warm_up_steps	plc/has_ReduceLROnPlateau	plc/has_learning_rate_decay	plc/lr_scheduler_frequency	plc/lr_scheduler_interval	plc/lr_scheduler_steps_frequency	plc/only_train_heads	plc/reduceLROnPlateau_config/cooldown	plc/reduceLROnPlateau_config/eps	plc/reduceLROnPlateau_config/factor	plc/reduceLROnPlateau_config/min_lr	plc/reduceLROnPlateau_config/patience	plc/reduceLROnPlateau_config/verbose	plc/regularization_coef	plc/regularize_from_init	plc/stepLR_gamma	plc/stepLR_step_size	plc/token_location	plc/train_all	pltc	pltc/accumulate_grad_batches	pltc/check_val_every_n_epoch	pltc/enable_checkpointing	pltc/limit_test_batches	pltc/limit_train_batches	pltc/limit_val_batches	pltc/log_every_n_steps	pltc/max_epochs	pltc/max_steps	pltc/max_time	pltc/min_epochs	pltc/min_steps	pltc/num_sanity_val_steps	pltc/overfit_batches	pltc/precision	pltc/val_check_interval	profiler	regularization_coef	regularize_from_init	sampling_method	shuffle_test	shuffle_train	test-LFB-AVG-mse/dataloader_idx_1	test-LFB-LAST-mse/dataloader_idx_1	test-MeanSquaredError	test-MeanSquaredError/dataloader_idx_1	test-MulticlassAccuracy	test-MulticlassAccuracy/dataloader_idx_0	test-commonsense-acc	test-commonsense-acc/dataloader_idx_0	test_acc	to_save_model	torch_float32_matmul_precision	train_datasets	train_loss	trainer/global_step	val-commonsense-acc	val_acc	validation-MeanSquaredError	validation-MulticlassAccuracy	validation-commonsense-acc
"""

# %%
"""
commonsense-test-MulticlassAccuracy	
commonsense-test-MulticlassAccuracy/dataloader_idx_0
test-MulticlassAccuracy	
test-MulticlassAccuracy/dataloader_idx_0	
test-commonsense-acc	
test-commonsense-acc/dataloader_idx_0
test_acc		

commonsense-validation-MulticlassAccuracy
commonsense-validation-label-MulticlassAccuracy 
val_acc
val-commonsense-acc
validation-MulticlassAccuracy
validation-commonsense-acc
"""

# %%
runs_df["cs_test_set_acc"] = np.nan
for column in [
    "commonsense-test-MulticlassAccuracy",
    "commonsense-test-label-MulticlassAccuracy",
    "commonsense-test-MulticlassAccuracy/dataloader_idx_0",
    "test-MulticlassAccuracy",
    "test-MulticlassAccuracy/dataloader_idx_0",
    "test-commonsense-acc",
    "test-commonsense-acc/dataloader_idx_0",
    "test_acc",
]:
    runs_df["cs_test_set_acc"] = runs_df["cs_test_set_acc"].fillna(runs_df[column])


# %%

columns_to_merge = [
    "commonsense-validation-MulticlassAccuracy",
    "commonsense-validation-label-MulticlassAccuracy",
    "val_acc",
    "val-commonsense-acc",
    "validation-MulticlassAccuracy",
    "validation-commonsense-acc",
]

runs_df["cs_hard_set_acc"] = np.nan

for column in columns_to_merge:
    runs_df["cs_hard_set_acc"] = runs_df["cs_hard_set_acc"].fillna(runs_df[column])

# %%
runs_df["model_path"] = runs_df["model_path"].fillna(runs_df["checkpoint"])

# %%
myview = runs_df[runs_df["_runtime"] > 300.0][[
    "model_path",
    "_runtime",
    "cs_hard_set_acc",
    "cs_test_set_acc"
]]
myview

# %%
# To csv:
runs_df.to_csv("project.csv")
# %%

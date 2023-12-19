# %%
from pandas import Series
from pandas import json_normalize
from pprint import pp
from time import strftime, gmtime, localtime
from datetime import datetime
from transformers import AutoConfig
from plotly import express as px
import numpy as np
import pandas as pd
import wandb

api = wandb.Api()

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
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
    data_list.append(s | c | {"name": run.name} | {'tags': run.tags})

runs_df = pd.DataFrame(data_list)

# %%
# Sort columns by name:
runs_df = runs_df.reindex(sorted(runs_df.columns), axis=1)
# %%
runs_df.to_csv("report/project_original.csv")

# %%
# Load from csv:
runs_df = pd.read_csv("report/project_original.csv")

# %%
# Include only those after 1 Sep 2023:
runs_df = runs_df[runs_df["_timestamp"] >= datetime(2023, 9, 1).timestamp()]
# %% 
# Get those items in runs_df that has 'bad-hyperparams' tag
runs_df = runs_df[runs_df['tags'].apply(lambda x: 'bad-hyperparams' not in x)]

# %%
# Clear extra quotes in 'checkpoint_path' and 'last_checkpoint_path' columns:
def clear_quotes(s):
    if s is None or not isinstance(s, str):
        return None
    return s.strip("\"'")


runs_df["checkpoint_path"] = runs_df["checkpoint_path"].apply(clear_quotes)
runs_df["last_checkpoint_path"] = runs_df["last_checkpoint_path"].apply(clear_quotes)

# %%
# Merge results for Commonsense Hard Test set:
runs_df["cs_hard_set_acc"] = np.nan
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
    runs_df["cs_hard_set_acc"] = runs_df["cs_hard_set_acc"].fillna(runs_df[column])

# %%
# Merge results for Commonsense Test set:
columns_to_merge = [
    "commonsense-validation-MulticlassAccuracy",
    "commonsense-validation-label-MulticlassAccuracy",
    "val_acc",
    "val-commonsense-acc",
    "validation-MulticlassAccuracy",
    "validation-commonsense-acc",
]
runs_df["cs_test_set_acc"] = np.nan
for column in columns_to_merge:
    runs_df["cs_test_set_acc"] = runs_df["cs_test_set_acc"].fillna(runs_df[column])

# %% 
# Remove rows with wrong accuracy, not in [0, 1] for cs_hard_set_acc and cs_test_set_acc:
runs_df = runs_df[runs_df["cs_hard_set_acc"].between(0, 1)]
runs_df = runs_df[runs_df["cs_test_set_acc"].between(0, 1)]
# %%
runs_df["model_path"] = runs_df["model_path"].fillna(runs_df["checkpoint"])

# %%
runs_df["timestamp"] = runs_df["_timestamp"].apply(
    lambda x: strftime("%Y-%m-%d %H:%M:%S %Z", gmtime(x))
)

# %% 
# Columns for what was a training dataset and if it was trained on:

# Check if in runs_df['ds1'] column (json string) has train.slicing != '[:0]':
def check_if_trained_on(ds_col):
    def check(row):
        ds = row[ds_col]
        if not isinstance(ds, dict):
            return None

        train_slicing = ds.get("train", {}).get("slicing", None)
        enable = ds.get("enable", None)
        if train_slicing is None or enable is None:
            return None
        return train_slicing != "[:0]" and enable

    return check


def get_ds_name(ds_col):
    def g(row):
        ds = row[ds_col]
        if isinstance(ds, dict):
            name = ds.get("name", "")
            if 'LFB' in name:
                return name.split('-')[0]
            return name
        return ""

    return g


for i, ds in enumerate(["ds1", "ds2"]):
    runs_df[ds] = runs_df.apply(get_ds_name(ds), axis=1)
    runs_df[ds] = runs_df[ds].fillna(
        runs_df["train_datasets"].apply(
            lambda x: x[i] if isinstance(x, list) and i < len(x) else None
        )
    )
    runs_df[f"{ds}_training"] = runs_df.apply(check_if_trained_on(ds), axis=1)

# %%
# Sampling method
runs_df["sampling_method"] = runs_df["sampling_method"].fillna(
    runs_df["ds2"].apply(
        lambda x: x["sampling_method"]
        if isinstance(x, dict) and "sampling_method" in x
        else (
            x["name"].split("-")[1]
            if isinstance(x, dict) and "name" in x and x["name"].startswith("LFB-")
            else None
        )
    )
)
# Remove 'Sampling.' prefix in 'sampfling_method' column:
runs_df["sampling_method"] = runs_df["sampling_method"].str.replace("Sampling.", "")

# %% 
# Find previous run ds1_training, ds2_training columns using its 'last_checkpoint_path' column that should match current row 'checkpoint_path' column:

def find_prev_run_id(row):
    prev_rows = runs_df[runs_df.index > row.name][
        runs_df["last_checkpoint_path"] == row["checkpoint_path"] 
    ]
    if len(prev_rows) > 0:
        # Return last row id in prev_rows:
        return int(prev_rows.index.min())
    return None

runs_df['prev_run_id'] = runs_df.apply(find_prev_run_id, axis=1)

def get_sequence(row):
    if row['prev_run_id'].is_integer():
        prev_run = runs_df.loc[int(row['prev_run_id'])]
        if prev_run.any():
            if (prev_run['ds1_training'] == True and row['ds2_training'] == True):
                return f"{prev_run['ds1']} then {prev_run['ds2']}"
            if (prev_run['ds2_training'] == True and row['ds1_training'] == True):
                return f"{prev_run['ds2']} then {prev_run['ds1']}"
    return "-"

runs_df['seq_train'] = runs_df.apply(get_sequence, axis=1)

# Get number of parameters for a HuggingFace model using 'model_path' field into 'parameters_num':
model_size_mln = {
    'bert-large-cased': 333,
    'bert-base-cased': 108,
    'roberta-large': 355,
    'microsoft/deberta-v2-xlarge': 884,

}
runs_df["model_size_mln"] = runs_df["model_path"].apply(
    lambda x: model_size_mln.get(x, None)
)

# %%
# To csv:
runs_df.to_csv("report/project.csv")

# %%
runs_df = pd.read_csv("report/project.csv")

# %%
# Create a view

# Filter out runs with with
myview = runs_df[
    [
        "model_path",
        "timestamp",
        "_runtime",
        "cs_hard_set_acc",
        "cs_test_set_acc",
        "last_checkpoint_path",
        "checkpoint_path",
        "_step",
    ]
]
bs_columns = [col for col in runs_df.columns if col.startswith("bs_")]
myview["bs_max"] = runs_df[bs_columns].max(axis=1)
myview["bs_std"] = runs_df[bs_columns].std(axis=1)
#cod_columns = [col for col in runs_df.columns if col.startswith("cod_")]
#myview["cod_median"] = runs_df[cod_columns].median(axis=1)
#myview["cod_std"] = runs_df[cod_columns].std(axis=1)
myview["steps"] = runs_df["_step"]
myview = myview.sort_values(by=["model_size_mln"])
myview
# %%
myview.to_csv("report/project_view.csv")


# %%
# Group by model_path, sampling_method, seq_train
by_model_sampling_seq = myview.groupby(["model_path", "model_size_mln", "sampling_method", "seq_train"]).agg(
    {
        "timestamp": ["count"],
        "cs_hard_set_acc": ["mean", "std", "max"],
        "cs_test_set_acc": ["mean", "std", "max"],
        "bs_max": ["median", "max"],
    }
)
# sort by model_size_mln:
by_model_sampling_seq = by_model_sampling_seq.sort_values(by=["model_size_mln"])

by_model_sampling_seq

# %%
by_model_sampling_seq.to_excel("report/by_model_sampling_seq.xlsx")

# %%
by_model_sampling_seq.plot.bar(
    y=[("cs_hard_set_acc", "mean")],
    yerr=[("cs_hard_set_acc", "std")],
    title="cs_hard_set_acc, cs_test_set_acc",
    # Make it wide:
    figsize=(10, 7),
)
# %%
by_model = myview.groupby(["model_path", "model_size_mln"]).agg(
    {
        "timestamp": ["count"],
        "cs_hard_set_acc": ["mean", "std", "max"],
        "cs_test_set_acc": ["mean", "std", "max"],
        "bs_max": ["median", "max"],
    }
)
display(by_model)
by_model.to_excel("report/by_model.xlsx")

# Plot cs_test_set_acc mean as line and std as error for each model with x axis as model_size_mln:
px.bar(
    x=by_model.index.get_level_values("model_size_mln"),
    y=by_model["cs_test_set_acc"]["mean"],
    error_y=by_model["cs_test_set_acc"]["std"],
    title="Commomnsense Test Set Accuracy",
    labels={"x": "Model Size (mln)", "y": "Accuracy"},
)

# %%
px.bar(
    x=by_model.index.get_level_values("model_size_mln"),
    y=by_model["cs_hard_set_acc"]["mean"],
    error_y=by_model["cs_hard_set_acc"]["std"],
    title="Commonsense Hard Test Set Accuracy",
    labels={"x": "Model Size (mln)", "y": "Accuracy"},
)
# %%

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
runs_data = api.runs("asdfasdfasdfdsafsd/AISC_BB")

summary_list, config_list, name_list = [], [], []
data_list = []
for run in runs_data:
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

rdf = pd.DataFrame(data_list)  # Runs DataFrame.

# %%
# Sort columns by name:
rdf = rdf.reindex(sorted(rdf.columns), axis=1)
# %%
rdf.to_csv("report/project_original.csv")

# %%
# Load from csv:
#rdf = pd.read_csv("report/project_original.csv")

# %%
# Include only those after 1 Sep 2023:
rdf = rdf[rdf["_timestamp"] >= datetime(2023, 9, 1).timestamp()]
# %% 
# Get those items in runs_df that has 'bad-hyperparams' tag
rdf = rdf[rdf['tags'].apply(lambda x: 'bad-hyperparams' not in x)]

# %%
# Clear extra quotes in 'checkpoint_path' and 'last_checkpoint_path' columns:
def clear_quotes(s):
    if s is None or not isinstance(s, str):
        return s
    return s.strip("\"'")


rdf["checkpoint_path"] = rdf["checkpoint_path"].apply(clear_quotes)
rdf["last_checkpoint_path"] = rdf["last_checkpoint_path"].apply(clear_quotes)

# %%
# Merge results for Commonsense Hard Test set:
rdf["cs_hard_set_acc"] = np.nan
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
    rdf["cs_hard_set_acc"] = rdf["cs_hard_set_acc"].fillna(rdf[column])

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
rdf["cs_test_set_acc"] = np.nan
for column in columns_to_merge:
    rdf["cs_test_set_acc"] = rdf["cs_test_set_acc"].fillna(rdf[column])

# %% 
# Remove rows with wrong accuracy, not in [0, 1] for cs_hard_set_acc and cs_test_set_acc:
#rdf = rdf[rdf["cs_hard_set_acc"].between(0, 1)]
rdf = rdf[rdf["cs_test_set_acc"].between(0, 1)]
# %%
rdf["model_path"] = rdf["model_path"].fillna(rdf["checkpoint"])

# %%
rdf["timestamp"] = rdf["_timestamp"].apply(
    lambda x: strftime("%Y-%m-%d %H:%M:%S %Z", gmtime(x))
)

# %% 
# Set ds1 and ds2 cols (what datasets were used):

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
    rdf[f"{ds}_name"] = rdf.apply(get_ds_name(ds), axis=1)
    rdf[f"{ds}_name"] = rdf[f"{ds}_name"].fillna(
        rdf["train_datasets"].apply(
            lambda x: x[i] if isinstance(x, list) and i < len(x) else None
        )
    )

# %%
# Set ds1/ds2_training columns (whether the model was trained on ds1/ds2):
def check_if_trained_on(ds_col):
    # Check if in runs_df['ds1'] column (json string) has train.slicing != '[:0]':
    def check(row):
        ds = row[ds_col]
        if not isinstance(ds, dict):
            return None

        slicing = ds.get("train", {}).get("slicing", None)
        enable = ds.get("enable", None)
        if slicing is None or enable is None:
            return None
        return slicing != "[:0]" and enable

    return check


for i, ds in enumerate(["ds1", "ds2"]):
    rdf[f"{ds}_training"] = rdf.apply(check_if_trained_on(ds), axis=1)

# %%
# Sampling method

def get_sampling_method(x):
    if isinstance(x, dict) and "sampling_method" in x:
        return x["sampling_method"]
    elif isinstance(x, dict) and "name" in x and x["name"].startswith("LFB-"):
        return x["name"].split("-")[1]
    else:
        return None

rdf["sampling_method"] = None
rdf["sampling_method"] = rdf["sampling_method"].fillna(rdf["ds2"].apply(get_sampling_method))
rdf["sampling_method"] = rdf["sampling_method"].fillna(rdf["ds2/sampling_method"])
# Remove 'Sampling.' prefix in 'sampfling_method' column:
rdf["sampling_method"] = rdf["sampling_method"].str.replace("Sampling.", "")
# Fill sampling_method col with None for NaN and empty string values:
rdf["sampling_method"] = rdf["sampling_method"].fillna('')


# %% 
# Find previous runs for each run:
def find_prev_run_id(row):
    prev_rows = rdf[rdf.index > row.name][
        rdf["last_checkpoint_path"] == row["checkpoint_path"] 
    ]
    if len(prev_rows) > 0:
        # Return last row id in prev_rows:
        return int(prev_rows.index.min())
    return None
rdf['prev_run_id'] = rdf.apply(find_prev_run_id, axis=1)

# %% 
# Set seq_train column (what datasets were used):
def get_sequence(row):
    if row['prev_run_id'].is_integer():
        prev_run = rdf.loc[int(row['prev_run_id'])]
        if prev_run.any():
            def append(ds, r):
                if r[f"{ds}_training"] == True:
                    return r[f"{ds}_name"]
                return None

            def row_seq(r):
                return [append('ds1', r), append('ds2', r)]
            
            seq = ' and '.join(e for e in row_seq(prev_run) if e is not None)
            seq += ' then '
            seq += ' and '.join(e for e in row_seq(row) if e is not None)
            return seq
    return ""
rdf['seq_train'] = rdf.apply(get_sequence, axis=1)

# %%
# Get number of parameters for a HuggingFace model using 'model_path' field into 'parameters_num':
model_size_mln = {
    'bert-large-cased': 333,
    'bert-base-cased': 108,
    'roberta-large': 355,
    'microsoft/deberta-v2-xlarge': 884,
}

rdf["model_size_mln"] = rdf["model_path"].apply(
    lambda x: model_size_mln.get(x, None)
)

# %% 
rdf['only_on_ethics'] = rdf['ds1_training'].fillna(False) & ~rdf['ds2_training'].fillna(False) & ~rdf['seq_train'].apply(lambda x: 'LFB' in x if isinstance(x, str) else False)

# %%
# To csv:
rdf.to_csv("report/project.csv")

# %%
# rdf = pd.read_csv("report/project.csv")

# %%
# Create a view

# Filter out runs with with
a_view = rdf[[
    "model_path",
    "timestamp",
    "_runtime",
    "cs_hard_set_acc",
    "cs_test_set_acc",
    "last_checkpoint_path",
    "checkpoint_path",
    "_step",
    "model_size_mln",
    "seq_train",
    "sampling_method",
    'only_on_ethics',
]]
# Drop rows without cs_hard_set_acc:
a_view = a_view.dropna(subset=["cs_hard_set_acc"])
# Drop rows without cs_test_set_acc:
a_view = a_view.dropna(subset=["cs_test_set_acc"])
bs_columns = [col for col in rdf.columns if col.startswith("bs_")]
a_view["bs_max"] = rdf[bs_columns].max(axis=1)
a_view["bs_std"] = rdf[bs_columns].std(axis=1)
#cod_columns = [col for col in runs_df.columns if col.startswith("cod_")]
#myview["cod_median"] = runs_df[cod_columns].median(axis=1)
#myview["cod_std"] = runs_df[cod_columns].std(axis=1)
a_view["steps"] = rdf["_step"]
a_view = a_view.sort_values(by=["model_size_mln"])
a_view
# %%
a_view.to_csv("report/project_view.csv")


# %%
# Group by model_path, sampling_method, seq_train
by_model_by_ethics = a_view.groupby(
    ["model_path", "model_size_mln", 'only_on_ethics']
).agg({
        "timestamp": ["count"],
        "cs_hard_set_acc": ["mean", "std", "max"],
        "cs_test_set_acc": ["mean", "std", "max"],
        #"bs_max": ["median", "max"],
})
# sort by model_size_mln:
by_model_by_ethics = by_model_by_ethics.sort_values(by=["model_size_mln"])

by_model_by_ethics

# %%
by_model_by_ethics.to_excel("report/by_model_by_ethics.xlsx")

# %%
by_model_by_ethics.plot.bar(
    y=("cs_test_set_acc", "max"),
    yerr=("cs_test_set_acc", "std"),
    title="Commomnsense Test Set Accuracy",
    ylabel="Accuracy",
    xlabel="Model Size (mln)",
    figsize=(20, 10),
)
# %%
by_model = a_view.groupby(["model_path", "model_size_mln"]).agg(
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

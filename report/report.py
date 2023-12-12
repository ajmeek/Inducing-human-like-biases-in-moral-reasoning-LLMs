# %%
import pandas as pd
from pandas import Series
import numpy as np
from pandas import json_normalize
import wandb
import json
from pprint import pp
from time import strftime, gmtime, localtime

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
    data_list.append(s | c | {"name": run.name})

runs_df = pd.DataFrame(data_list)

# %%
# Sort columns by name:
runs_df = runs_df.reindex(sorted(runs_df.columns), axis=1)
# %%
runs_df.to_csv("report/project_original.csv")

# %%
runs_df = pd.load_csv("project_original.csv")

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
runs_df["model_path"] = runs_df["model_path"].fillna(runs_df["checkpoint"])

# %%
# To csv:
runs_df.to_csv("report/project.csv")

# %%
runs_df = pd.load_csv("project.csv")

# %%
# Create view:

# Filter out runs with with 
myview = runs_df[
    ["model_path", "_runtime", "cs_hard_set_acc", "cs_test_set_acc"]
]

myview["timestamp"] = runs_df["_timestamp"].apply(
    lambda x: strftime("%Y-%m-%d %H:%M:%S %Z", gmtime(x))
)
bs_columns = [col for col in runs_df.columns if col.startswith("bs_")]
myview["bs_median"] = runs_df[bs_columns].median(axis=1)
myview["bs_std"] = runs_df[bs_columns].std(axis=1)
cod_columns = [col for col in runs_df.columns if col.startswith("cod_")]
myview["cod_median"] = runs_df[cod_columns].median(axis=1)
myview["cod_std"] = runs_df[cod_columns].std(axis=1)
myview["checkpoint_path"] = runs_df["checkpoint_path"]
myview["steps"] = runs_df["_step"]


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
            return ds.get("name", "")
        return ""

    return g


for i, ds in enumerate(["ds1", "ds2"]):
    myview[ds] = runs_df.apply(get_ds_name(ds), axis=1)
    myview[f"{ds}_training"] = runs_df.apply(check_if_trained_on(ds), axis=1)
    myview[ds] = myview[ds].fillna(
        runs_df["train_datasets"].apply(
            lambda x: x[i] if isinstance(x, list) and i < len(x) else None
        )
    )

myview["sampling_method"] = runs_df["sampling_method"]
myview["sampling_method"] = myview["sampling_method"].fillna(
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
myview["sampling_method"] = myview["sampling_method"].str.replace("Sampling.", "")

# Filter out dirty runs:
myview = runs_df[runs_df["_runtime"] > 300.0][

myview

# %%
by_model_path = (
    myview[
        (myview["ds1_training"].isnull() | myview["ds1_training"])
        & (myview["ds2_training"].isnull() | myview["ds2_training"])
    ][
        [
            "model_path",
            "_runtime",
            "cs_hard_set_acc",
            "cs_test_set_acc",
            "bs_median",
            "bs_std",
            "cod_median",
            "cod_std",
            "steps",
            "sampling_method",
        ]
    ]
    .groupby(["model_path", "sampling_method"])
    .agg(
        {
            "_runtime": ["sum"],
            "cs_hard_set_acc": ["mean", "std", "max"],
            "cs_test_set_acc": ["mean", "std", "max"],
            #"bs_median": ["mean", "std"],
            #"bs_std": ["mean", "std"],
            #"cod_median": ["mean", "std"],
            #"cod_std": ["mean", "std"],
            "steps": ["sum"],
        }
    )
    .reset_index()
)

by_model_path["num_runs"] = (
    myview.groupby(["model_path", "sampling_method"])
    .size()
    .reset_index(name="num_runs")["num_runs"]
)
# Move 'num_runs' column to the beginning:
cols = list(by_model_path.columns)
cols = [cols[-1]] + cols[:-1]
by_model_path = by_model_path[cols]
display(by_model_path)
# %%

# %%
myview.to_csv("report/project_view.csv")

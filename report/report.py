#
# Generates report for fine tuning experiments.

# %%
from time import strftime, gmtime
from datetime import datetime
from plotly import express as px
from functools import partial
import numpy as np
import pandas as pd
import wandb

api = wandb.Api()

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
# %%
# Load data from WANDB
runs_data = api.runs("asdfasdfasdfdsafsd/AISC_BB")
summary_list, config_list, name_list = [], [], []
data_list = []
for run in runs_data:
    if "_runtime" not in run.summary:
        continue
    # .summary contains the output keys/values for metrics like accuracy.
    #          We call ._json_dict to omit large files
    # .config contains the hyperparameters.
    #         We remove special values that start with _.
    # .name is the human-readable name of the run.
    s = run.summary._json_dict
    c = {k: v for k, v in run.config.items() if not k.startswith("_")}
    data_list.append(s | c | {"name": run.name} | {"tags": run.tags})

rdf = pd.DataFrame(data_list)  # Runs DataFrame.
rdf = rdf.reindex(sorted(rdf.columns), axis=1)
# %%
rdf.to_csv("report/project_original.csv")

# %%
# Load from csv:
# rdf = pd.read_csv("report/project_original.csv")

# %%
# Only those after 1 Sep 2023 because we didn't train much and there were some bugs.
rdf = rdf[rdf["_timestamp"] >= datetime(2023, 9, 1).timestamp()]
# %%
rdf = rdf[rdf["tags"].apply(lambda x: "bad-hyperparams" not in x)]


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
rdf["cs_test_set_acc"] = np.nan
for column in [
    "commonsense-validation-MulticlassAccuracy",
    "commonsense-validation-label-MulticlassAccuracy",
    "val_acc",
    "val-commonsense-acc",
    "validation-MulticlassAccuracy",
    "validation-commonsense-acc",
]:
    rdf["cs_test_set_acc"] = rdf["cs_test_set_acc"].fillna(rdf[column])

# %%
# Remove rows with wrong or no accuracy:
# rdf = rdf[rdf["cs_hard_set_acc"].between(0, 1)]
rdf = rdf[rdf["cs_test_set_acc"].between(0, 1)]
# %%
rdf["model_path"] = rdf["model_path"].fillna(rdf["checkpoint"])

# %%
rdf["timestamp"] = rdf["_timestamp"].apply(
    lambda x: strftime("%Y-%m-%d %H:%M:%S %Z", gmtime(x))
)

# %%
# Set ds1_name / ds2_name columns (what datasets were used):


def get_ds_name(row, ds_col, remove_sampling=True):
    ds = row[ds_col]
    name = ""
    if isinstance(ds, dict):
        name = ds.get("name", "")
    elif row[f"{ds_col}/name"]:
        name = row[f"{ds_col}/name"]
    # Correct:
    if "LFB-" in name and remove_sampling:
        name = name.split("-")[0]
    if name == "learning_from_brains":
        name = "LFB"
    return name


for i, ds in enumerate(["ds1", "ds2"]):
    rdf[f"{ds}_name"] = rdf.apply(partial(get_ds_name, ds_col=ds), axis=1)
    rdf[f"{ds}_name"] = rdf[f"{ds}_name"].fillna(
        rdf["train_datasets"].apply(
            lambda x: x[i] if isinstance(x, list) and i < len(x) else None
        )
    )


# %%
# Set ds1/ds2_training columns (whether the model was trained on ds1/ds2):
def check_if_trained_on(ds_col):
    def check(row):
        ds = row[ds_col]
        slicing = None
        enable = None
        if isinstance(ds, dict):
            slicing = ds.get("train", {}).get("slicing", None)
            enable = ds.get("enable", None)
        elif slicing is None or enable is None:
            enable = row[f"{ds_col}/enable"]
            slicing = row[f"{ds_col}/train/slicing"]
        if slicing is None or enable is None:
            return None
        return slicing != "[:0]" and enable

    return check


for i, ds in enumerate(["ds1", "ds2"]):
    rdf[f"{ds}_training"] = rdf.apply(check_if_trained_on(ds), axis=1)
    rdf[f"{ds}_training"] = rdf[f"{ds}_training"].fillna(True)

# %%
# Sampling method


def get_sampling_method(row):
    sm = ""
    name = get_ds_name(row, ds_col="ds2", remove_sampling=False)
    if "-" in name:
        sm = name.split("-")[1] if "-" in name else None
    elif "sampling_method" in row and row["sampling_method"]:
        sm = row["sampling_method"]
    elif "ds2/sampling_method" in row and row["ds2/sampling_method"]:
        sm = row["ds2/sampling_method"]
    if isinstance(sm, str):
        return sm.replace("Sampling.", "")
    return sm


rdf["ds2/sampling_method"] = rdf["ds2/sampling_method"].fillna("")
rdf["sampling_method"] = rdf["sampling_method"].fillna("")
rdf["sampling_method"] = rdf.apply(get_sampling_method, axis=1)


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


rdf["prev_run_id"] = rdf.apply(find_prev_run_id, axis=1)


# %%
# Set seq_train column (what datasets were used):
def get_sequence(row):
    if row["prev_run_id"].is_integer():
        prev_run = rdf.loc[int(row["prev_run_id"])]
        if prev_run.any():

            def append(ds, r):
                if r[f"{ds}_training"] == True:
                    return r[f"{ds}_name"]
                return None

            def row_seq(r):
                return [append("ds1", r), append("ds2", r)]

            seq = " and ".join(e for e in row_seq(prev_run) if e is not None)
            seq += " then "
            seq += " and ".join(e for e in row_seq(row) if e is not None)
            return seq
    return ""


rdf["seq_train"] = rdf.apply(get_sequence, axis=1)

# %%
# Get number of parameters for a HuggingFace model using 'model_path' field into 'parameters_num':
model_size_mln = {
    "bert-large-cased": 333,
    "bert-base-cased": 108,
    "roberta-large": 355,
    "microsoft/deberta-v2-xlarge": 884,
}

rdf["model_size_mln"] = rdf["model_path"].apply(lambda x: model_size_mln.get(x, None))

# %%
# Set only_on_ethics column (whether the model was trained only on ethics).
# Assume ds1 is ethics and ds2 is brain data.

rdf["only_on_ethics"] = (
    (rdf["ds1_training"] == True)
    & (rdf["ds1_name"] == "commonsense")
    & (rdf["ds2_training"] == False)
    & (rdf["ds2_name"] != "commonsense")
    # Previous not trained on LFB:
    & ~(rdf["seq_train"].apply(lambda x: "LFB" in x if isinstance(x, str) else False))
)

# %%
rdf["only_on_lfb"] = (
    (rdf["ds1_training"] == False)
    & (rdf["ds1_name"] == "commonsense")
    & (rdf["ds2_training"] == True)
    & (rdf["ds2_name"] != "commonsense")
    # Previous not trained on LFB:
    & ~(
        rdf["seq_train"].apply(
            lambda x: "commonsense" in x if isinstance(x, str) else False
        )
    )
)

# %%
# To csv:
rdf.to_csv("report/project.csv")

# %%
# rdf = pd.read_csv("report/project.csv")

# %%
# Create a view

# Filter out runs with with
a_view = rdf[
    [
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
        "only_on_ethics",
        "only_on_lfb",
    ]
]
# Drop rows without cs_hard_set_acc:
a_view = a_view.dropna(subset=["cs_hard_set_acc"])
# Drop rows without cs_test_set_acc:
a_view = a_view.dropna(subset=["cs_test_set_acc"])
bs_columns = [col for col in rdf.columns if col.startswith("bs_")]
a_view["bs_max"] = rdf[bs_columns].max(axis=1)
a_view["bs_std"] = rdf[bs_columns].std(axis=1)
# cod_columns = [col for col in runs_df.columns if col.startswith("cod_")]
# myview["cod_median"] = runs_df[cod_columns].median(axis=1)
# myview["cod_std"] = runs_df[cod_columns].std(axis=1)
a_view["steps"] = rdf["_step"]
a_view = a_view.sort_values(by=["model_size_mln"])
a_view

# %%
a_view.to_csv("report/project_view.csv")


# %%
# Group by model_path, sampling_method, seq_train
by_m_e = (
    a_view[~a_view["only_on_lfb"]]
    .groupby(["model_path", "model_size_mln", "only_on_ethics"])
    .agg(
        {
            "timestamp": ["count"],
            "cs_hard_set_acc": ["mean", "std", "max"],
            "cs_test_set_acc": ["mean", "std", "max"],
            # "bs_max": ["median", "max"],
        }
    )
)
# sort by model_size_mln:
by_m_e = by_m_e.sort_values(by=["model_size_mln"])

# %%
# Ungroup by_model_by_ethics into a flat table:
by_m_e_flat = by_m_e.reset_index()
by_m_e_flat.columns = [
    "model_path",
    "model_size_mln",
    "only_on_ethics",
    "runs",
    "cs_hard_set_acc",
    "cs_hard_set_acc_std",
    "cs_hard_set_acc_max",
    "cs_test_set_acc",
    "cs_test_set_acc_std",
    "cs_test_set_acc_max",
]
by_m_e_flat["only_on_ethics"] = by_m_e_flat["only_on_ethics"].apply(
    lambda x: "Only Ethics" if x else "On LFB"
)

# %%
# Create bar for Commonsense Test set accuracy for each model_path:
fig = px.bar(
    by_m_e_flat,
    x="model_path",
    y="cs_test_set_acc",
    error_y="cs_test_set_acc_std",
    color="only_on_ethics",
    title="Commomnsense Accuracy (Test Set)",
    labels={
        "x": "Model Size (mln params)",
        "y": "Accuracy",
        "only_on_ethics": "Training",
        "cs_test_set_acc": "Accuracy",
        "model_path": "Model",
    },
    barmode="group",
    text_auto=True,
)
fig.update_layout(yaxis_tickformat=".2%")
fig.show()

# %%
# Create bar for Commonsense Hard set accuracy for each model_path:
fig = px.bar(
    by_m_e_flat,
    x="model_path",
    y="cs_hard_set_acc",
    error_y="cs_hard_set_acc_std",
    color="only_on_ethics",
    title="Commomnsense Accuracy (Hard Set)",
    labels={
        "x": "Model Size (mln params)",
        "y": "Accuracy",
        "only_on_ethics": "Training",
        "cs_hard_set_acc": "Accuracy",
        "model_path": "Model",
    },
    barmode="group",
    text_auto=True,
)
fig.update_layout(yaxis_tickformat=".2%")
fig.show()

# %%
# Create table for the report:
by_m_e = by_m_e.reset_index()
# %% 
# Join mean and std columns and convert to percentage:
def format_acc(row, col_name):
    return f"{row[col_name, 'mean']:.1%} ± {row[col_name, 'std']:.1%}"

for cn in ["cs_hard_set_acc", "cs_test_set_acc"]:
    by_m_e[(cn, 'mean')] = by_m_e.apply(partial(format_acc, col_name=cn), axis=1)
    by_m_e[(cn, 'max')] = by_m_e.apply(lambda row: f"{row[cn, 'max']:.1%}", axis=1)


# %%
# Rename columsn, format values:
# Delete (cs_hard_set_acc,std) and (cs_test_set_acc,std) columns:
del by_m_e["cs_hard_set_acc", "std"]
del by_m_e["cs_test_set_acc", "std"]

by_m_e["only_on_ethics"] = by_m_e["only_on_ethics"].apply(lambda x: "✓" if x else "")
by_m_e.rename(
    columns={
        "timestamp": "Runs",
        "model_path": "Model",
        "model_size_mln": "Params, mln",
        "only_on_ethics": "On Ethics only",
        "cs_hard_set_acc": "Commonsense Hard Set",
        "cs_test_set_acc": "Commonsense Test Set",
    },
    inplace=True,
)

# %% 
# Sort columns:
by_m_e = by_m_e[
    [
        "Model",
        "Params, mln",
        "On Ethics only",
        "Runs",
        "Commonsense Hard Set",
        "Commonsense Test Set",
    ]
]
by_m_e

# %%
by_m_e.to_excel("report/by_model_ethics.xlsx")

# %%
# Group by model_path, sampling_method, take trained on LFB
by_m_sm = (
    a_view[~a_view["only_on_ethics"]]
    .groupby(["model_path", "model_size_mln", "sampling_method"])
    .agg(
        {
            "timestamp": ["count"],
            "cs_hard_set_acc": ["mean", "std", "max"],
            "cs_test_set_acc": ["mean", "std", "max"],
        }
    )
)
# sort by model_size_mln:
by_m_sm = by_m_sm.sort_values(by=["model_size_mln"])

# %%
# Ungroup by_model_by_ethics into a flat table:
by_m_sm_flat = by_m_sm.reset_index()
by_m_sm_flat.columns = [
    "model_path",
    "model_size_mln",
    "sampling_method",
    "runs",
    "cs_hard_set_acc",
    "cs_hard_set_acc_std",
    "cs_hard_set_acc_max",
    "cs_test_set_acc",
    "cs_test_set_acc_std",
    "cs_test_set_acc_max",
]

# %%
# Create bar for Commonsense Test set accuracy for each model_path:
fig = px.bar(
    by_m_sm_flat,
    x="model_path",
    y="cs_test_set_acc",
    error_y="cs_test_set_acc_std",
    color="sampling_method",
    title="Commomnsense Accuracy (Test Set)",
    labels={
        "x": "Model Size (mln params)",
        "y": "Accuracy",
        "sampling_method": "Sampling",
        "cs_test_set_acc": "Accuracy",
        "model_path": "Model",
    },
    barmode="group",
    text_auto=True,
)
fig.update_layout(yaxis_tickformat=".2%")
fig.show()

# %%
# Create bar for Commonsense Hard set accuracy for each model_path:
fig = px.bar(
    by_m_sm_flat,
    x="model_path",
    y="cs_hard_set_acc",
    error_y="cs_hard_set_acc_std",
    color="sampling_method",
    title="Commomnsense Accuracy (Hard Set)",
    labels={
        "x": "Model Size (mln params)",
        "y": "Accuracy",
        "sampling_method": "Sampling",
        "cs_hard_set_acc": "Accuracy",
        "model_path": "Model",
    },
    barmode="group",
    text_auto=True,
)
fig.update_layout(yaxis_tickformat=".2%")
fig.show()

# %%
# Create table for the report:
by_m_sm = by_m_sm.reset_index()
# %% 
# Join mean and std columns and convert to percentage:
def format_acc(row, col_name):
    return f"{row[col_name, 'mean']:.1%} ± {row[col_name, 'std']:.1%}"

for cn in ["cs_hard_set_acc", "cs_test_set_acc"]:
    by_m_sm[(cn, 'mean')] = by_m_sm.apply(partial(format_acc, col_name=cn), axis=1)
    by_m_sm[(cn, 'max')] = by_m_sm.apply(lambda row: f"{row[cn, 'max']:.1%}", axis=1)


# %%
# Rename columsn, format values:
# Delete (cs_hard_set_acc,std) and (cs_test_set_acc,std) columns:
del by_m_sm["cs_hard_set_acc", "std"]
del by_m_sm["cs_test_set_acc", "std"]

by_m_sm.rename(
    columns={
        "timestamp": "Runs",
        "model_path": "Model",
        "model_size_mln": "Params, mln",
        "only_on_ethics": "On Ethics only",
        "cs_hard_set_acc": "Commonsense Hard Set",
        "cs_test_set_acc": "Commonsense Test Set",
        "sampling_method": "Sampling"
    },
    inplace=True,
)


# %% 
# Sort columns:
by_m_sm = by_m_sm[
    [
        "Model",
        "Params, mln",
        "Sampling",
        "Runs",
        "Commonsense Hard Set",
        "Commonsense Test Set",
    ]
]
by_m_sm

# %%
by_m_sm.to_excel("report/by_model_sampling.xlsx")
# %%

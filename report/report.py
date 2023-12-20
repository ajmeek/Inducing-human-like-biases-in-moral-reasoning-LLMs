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
by_model_by_ethics = (
    a_view[~a_view["only_on_lfb"]]
    .groupby(['model_path', "model_size_mln", "only_on_ethics"])
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
by_model_by_ethics = by_model_by_ethics.sort_values(by=["model_size_mln"])
#by_model_by_ethics.rename(
#    columns={
#        "model_path": "Model",
#        "cs_hard_set_acc": "Commonsense Hard",
#        "cs_test_set_acc": "Commonsense Test",
#        "model_size_mln": "Size (mln params)",
#        "timestamp": "Runs",
#    },
#    inplace=True,
#)
# TODO: show std as "1.23±0.12"
by_model_by_ethics

# %%
by_model_by_ethics.to_excel("report/by_model_by_ethics.xlsx")

# %%
# Ungroup by_model_by_ethics into a flat table:
by_model_by_ethics = by_model_by_ethics.reset_index()
# %%
# Convert multiindex to columns:
by_model_by_ethics.columns = [
    "_".join(col).strip() for col in by_model_by_ethics.columns.values
]

# %%
# Using plotly, plot bar chart with x axis as model_size_mln and y axis as cs_test_set_acc mean and std,
# with same model_size_mln placed near each other:

fig = px.bar(
    x=by_model_by_ethics["model_size_mln_"],
    y=by_model_by_ethics["cs_test_set_acc_mean"],
    error_y=by_model_by_ethics["cs_test_set_acc_std"],
    color=by_model_by_ethics["only_on_ethics_"],
    title="Commomnsense Test Set Accuracy",
    labels={"x": "Model Size (mln)", "y": "Accuracy"},
    barmode="group",
    text=by_model_by_ethics["timestamp_count"],
    text_auto=True,
)
fig.update_traces(texttemplate="%{text:.2s}", textposition="outside", cliponaxis=True)
fig.show()


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

# %%
import pandas as pd 
import wandb
import json
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("asdfasdfasdfdsafsd/AISC_BB")

summary_list, config_list, name_list = [], [], []
for run in runs: 
    if '_runtime' not in run.summary:
        continue
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
          if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
    })

# runs_df.to_csv("project.csv")
# %%
for summary, config, name in zip(summary_list, config_list, name_list):
    if summary._runtime:
        print(summary._runtime)

# %%
# robert-large

# fmri and ethics

# %%
rows = [
    i for i, (s, c, n) in enumerate(zip(summary_list, config_list, name_list))
    if s['_ru']
]

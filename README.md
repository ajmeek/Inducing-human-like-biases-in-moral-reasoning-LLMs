# Inducing human-like biases in moral reasoning LMs 

This project is about fine-tuning language models (LMs) on a publicly available moral reasoning neuroimaging (fMRI) dataset, with the hope/expectation that this could help induce more human-like biases in the moral reasoning processes of LMs.

## How to run 

- `bash run.sh local` to provision local environment.
- `bash run.sh datasets [argument ..]` to prepare datasets. 
- `bash run.sh train [argument ..]` to train model and measure it. See `bash run.sh train --help` for the list of possible arguments.

# Inducing human-like biases in moral reasoning LMs 

This project is about fine-tuning language models (LMs) on a publicly available moral reasoning neuroimaging (fMRI) dataset, with the hope/expectation that this could help induce more human-like biases in the moral reasoning processes of LMs.

## How to run 

Provision using requirements.txt (pip) or environment(-cuda or -cpu).yml with Conda or Mamba. (See run.sh for how it can be done.)

Usage: `run.sh <function> [parameters...]`. Provide `--help` to see the list of possible arguments.

- `bash run.sh datasets [argument ..]` to prepare datasets. 
- `bash run.sh train [argument ..]` to train model and measure it.

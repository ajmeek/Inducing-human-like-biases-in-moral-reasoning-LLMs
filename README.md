# Inducing human-like biases in moral reasoning LMs 

This project is about fine-tuning language models (LMs) on a publicly available moral reasoning neuroimaging (fMRI) dataset, with the hope/expectation that this could help induce more human-like biases in the moral reasoning processes of LMs.

## How to run 

- `bash run.sh local` to provision local environment and activate it for current shell.
- `bash run.sh datasets [argument ..]` to create datasets. 
- `bash run.sh train [argument ..]` to train. See `bash run.sh train --help` for the list of possible arguments.
 
## Example of usage

To train in local environment:

```
bash run.sh local                   # to provision environment (conda or pip)
conda activate brain_bias           # if you use conda
bash run.sh datasets ds000212_raw   # to prepare datasets
bash run.sh train                   # to train a model and measure it
```

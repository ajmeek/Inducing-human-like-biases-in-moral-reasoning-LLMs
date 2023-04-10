# Inducing-human-like-biases-in-moral-reasoning-LLMs
This is a project for the 8th AI Safety Camp (AISC)
It includes the ETHICS dataset project: https://github.com/hendrycks/ethics

### Setup
Conda based setup: `conda create -f environment.yml`  
Pip based setup: `pip install -r requirements.txt` (not tested)

### How to run
Run `bin/train.sh` for the main script which fine-tunes BERT on a given dataset.
With the argument `--train_datasets` you can specify the datasets to train on.
It supports training on multiple datasets at once. The current options are: `ethics` and `ds000212`. If you specify both it will on each train step sample a batch from each dataset and train on it.
The default test set is ETHICS commonsense. And the classification head that
was used to train ETHICS will be saved to also test on the ETHICS commonsense dataset.

### How to train and evaluate on the ETHICS dataset (commonsense):
1. Download the dataset from https://github.com/hendrycks/ethics and put the csv files of the commonsense dataset ( csv has prefix 'cm')
in the `data/ethics/commonsense` folder in the git repository.
2. Run `data/ethics/commonsense/tune.py` to train and evaluate the model on ETHICS commonsense dataset.

Relevant parameters
- `batch_size` if forward pass makes you run out of GPU memory, reduce this. Minimum is 2 (1 doesn't work because of small bug)
- `ntest` number of test examples to use. I set it for 100 to quickly test (dataset is 3964 and 3885)
- `ntrain` number of training examples to use. I set it for 100 to quickly test (dataset is 13910)
- `model` model to run. 

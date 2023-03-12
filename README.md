# Inducing-human-like-biases-in-moral-reasoning-LLMs
This is a project for the 8th AI Safety Camp (AISC)
It includes the ETHICS dataset project: https://github.com/hendrycks/ethics

### Setup
Conda based setup: `conda create -f environment.yml`  
Pip based setup: `pip install -r requirements.txt` (not tested)

### How to run
Run `main.py` for the main script which now only executes a check for GPU and test
if hugging face is correctly installed by doing sentiment analysis using BERT.

### How to train and evaluate on the ETHICS dataset (commonsense):
1. Download the dataset from https://github.com/hendrycks/ethics and put the csv files of the commonsense dataset ( csv has prefix 'cm')
in the `ethics/commonsense` folder in the git repository.
2. Run `ethics/commonsense/tune.py` to train and evaluate the model on ETHICS commonsense dataset.

Relevant parameters
- `batch_size` if forward pass makes you run out of GPU memory, reduce this. Minimum is 2 (1 doesn't work because of small bug)
- `ntest` number of test examples to use. I set it for 100 to quickly test (dataset is 3964 and 3885)
- `ntrain` number of training examples to use. I set it for 100 to quickly test (dataset is 13910)
- `model` model to run. 
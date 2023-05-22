from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig, RobertaModel

from src.model import BERT
from src.utils.loading_data import load_ds000212_dataset

from sklearn.linear_model import RidgeCV


def calculate_brain_scores(model: nn.Module,
                           model_inputs: torch.tensor,
                           test_data: torch.tensor):

    activations = {}
    layers = ['23']
    modules = ['output.dense']
    all_layer_names = []
    for layer in layers:
        for module in modules:
            all_layer_names.append(f'{layer}.{module}')

    def get_activation(name):
        def hook(intermediate_model, input, output):
            if any(x in name for x in all_layer_names):
                activations[name] = output.detach()

        return hook

    # Attach the hook to every layer
    for name, layer in model.named_modules():
        layer.register_forward_hook(get_activation(name))

    # Run the model
    model.eval()
    with torch.no_grad():
        tokens, attention_mask = model_inputs[0], model_inputs[1]
        model(tokens, attention_mask)

    # Calculate the brain scores
    brain_scores = {}
    for layer_name, activation in activations.items():
        print('Calculating brain score for layer:', layer_name,
              'and activation dims: ', activation.shape, '...')
        activations_flattened = activation.reshape(activation.shape[0], -1)

        part_activations_flattened = activations_flattened
        part_test_data = test_data[:, :50]

        part_activations_flattened = part_activations_flattened.\
            repeat(activation.shape[0], 1)
        part_test_data = part_test_data.\
            repeat_interleave(activation.shape[0], dim=0)
        clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).\
            fit(part_activations_flattened, part_test_data)
        brain_scores[layer_name] = clf.score(part_activations_flattened,
                                             part_test_data)

    return brain_scores


if __name__ == '__main__':
    # Specify parameters
    path_to_model = r'..\models\cm_roberta-large.pt'  # Specify the path to the model.
    checkpoint_name = 'roberta-large'  # Specify the checkpoint name of the model. 'bert-base-cased' | 'roberta-large'
    train_head_dims = [2, 39127]  # Need to fill this in to not get an error when loading the model. This is not used in the brain score calculation.

    # Load the model
    # model = AutoModel.from_pretrained(checkpoint_name)
    # # model = BERT(model, head_dims=train_head_dims)
    # model.load_state_dict(torch.load(path_to_model))

    config = AutoConfig.from_pretrained('roberta-large', num_labels=1)
    model = RobertaModel.from_pretrained('../models/cm_roberta-large.pt', local_files_only=True, config=config)
    # for name, layer in model.named_modules():
    #     print(name)

    # Load the data
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)

    test_data_path = Path(r'..\data')
    fmri_data = load_ds000212_dataset(test_data_path, tokenizer, num_samples=999999, normalize=True)
    test_data = fmri_data[2]
    model_inputs = fmri_data[:2]
    # Calculate the brain scores
    brain_scores = calculate_brain_scores(model, model_inputs, test_data)
    print(brain_scores)

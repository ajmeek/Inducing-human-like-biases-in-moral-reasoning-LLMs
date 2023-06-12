from os import environ
from pathlib import Path, WindowsPath
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, RobertaModel

from src.main import get_config
from src.model import BERT
from src.utils.loading_data import load_ds000212

from sklearn.linear_model import RidgeCV


def calculate_brain_scores(model: nn.Module,
                           model_inputs: torch.tensor,
                           test_data: torch.tensor,
                           layers: list,
                           modules: list,
                           max_fmri_features: Optional[int]):
    activations = {}
    all_layer_names = []
    for layer in layers:
        for module in modules:
            all_layer_names.append(f'{layer}.{module}')

    def get_activation(name):
        def hook(model, input, output):
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
        # activations_flattened = activation.reshape(activation.shape[0], -1)
        activations_indices_last_token = torch.sum(model_inputs[1], dim=1) - 1

        index_tensor = activations_indices_last_token.long()

        # We need to create an additional dimension for the index tensor
        index_tensor = index_tensor.view(-1, 1, 1)

        # Now we can gather along the second dimension
        activations_last_token = torch.gather(activation, 1, index_tensor.expand(-1, -1, activation.shape[-1])).squeeze(1)

        if max_fmri_features is not None:
            test_data = test_data[:, :max_fmri_features]

        clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(activations_last_token,
                                                        test_data)
        brain_scores[layer_name] = clf.score(activations_last_token, test_data)

    return brain_scores


if __name__ == '__main__':
    # Note, this is expected to be run from the root of the project. (not src).
    # In Pycharm, can click top-right left of the run button, expand dropdown menu, click on three dots next to calculate_brain_scores, click Edit
    # and set the working directory to the root of the project.
    # Specify parameters
    path_to_model = r'models/cm_roberta-large.pt'  # Specify the path to the model.
    checkpoint_name = 'roberta-large'  # Specify the checkpoint name of the model. 'bert-base-cased' | 'roberta-large'

    # Load our custom pre-trained model on ETHICS and fMRI data.
    train_head_dims = [2, 39127]  # Need to fill this in to not get an error when loading the model. This is not used in the brain score calculation.
    # model = AutoModel.from_pretrained(checkpoint_name)
    # # model = BERT(model, head_dims=train_head_dims)
    # model.load_state_dict(torch.load(path_to_model))

    # Load Roberta model from local files.
    model_config = AutoConfig.from_pretrained('roberta-large', num_labels=1)
    model = RobertaModel.from_pretrained(path_to_model, local_files_only=True, config=model_config)

    # for name, layer in model.named_modules():
    #     print(name)

    # Load the data
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
    test_data_path = Path(environ.get('AISCBB_DATA_DIR','./data'))
    config = get_config()
    config['batch_size'] = 1000  # Make the batch large enough so we definitely have one subject. This is a bit hacky but works for now.
    fmri_data = load_ds000212(test_data_path, tokenizer, config, subject='sub-03')
    data = iter(fmri_data[0]).next()  # Get the first batch of data which is one entire subject.
    model_inputs = (data[0], data[1])
    test_data = data[2]

    # Calculate the brain scores
    layers = ['23']   # The layers to calculate the brain scores for.
    modules = ['output.dense']  # The layer and module will be combined to 'layer.module' to get the activations.
    max_fmri_features = None  # This is used to limit the size of the data so that everything can still fit in memory. If None, all features are used.
    brain_scores = calculate_brain_scores(model, model_inputs, test_data, layers, modules, max_fmri_features)
    print(brain_scores)

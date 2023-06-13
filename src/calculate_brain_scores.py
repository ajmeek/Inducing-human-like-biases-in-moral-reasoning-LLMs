import csv
import os
from datetime import datetime
from os import environ
from pathlib import Path, WindowsPath
from typing import Optional

import pandas as pd
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
                           max_fmri_features: Optional[int],
                           score_per_feature: bool = False,
                           train_perc : float = 1.0,
                           val_perc : float = 0.0,
                           ) -> dict:
    """
    Calculates the brain scores for the given model and test data.
    The activations of the model are calculated for the given layers and modules.
    The brain scores are calculated by fitting a Ridge regression model to the activations and the fMRI data.

    :param model: The model that does the forward pass
    :param model_inputs: The scenario data and input to the model.
    :param test_data: The fMRI data.
    :param layers: The layers for which the activations should be calculated.
    :param modules: The modules for which the activations should be calculated.
    :param max_fmri_features: For memory reasons, the number of fMRI features can be limited.
    :param score_per_feature: If true, the brain scores are calculated for each fMRI feature separately.
    :param train_perc: The percentage of the data to use for training.
    :param val_perc: The percentage of the data to use for validation.
    :return: A dictionary containing the brain scores for each layer and module.
    """
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
        print('Running model on test data...')
        tokens, attention_mask = model_inputs[0], model_inputs[1]
        # Flatten first two dimensions
        tokens = tokens.view(-1, tokens.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        model(tokens, attention_mask)

    # Calculate the brain scores
    brain_scores = {'layer.module': [], 'brain_score': []}
    for layer_name, activation in activations.items():
        print('Calculating brain score for layer:', layer_name,
              'and activation dims: ', activation.shape, '...')
        # activations_flattened = activation.reshape(activation.shape[0], -1)
        activations_indices_last_token = torch.sum(attention_mask, dim=1) - 1  # Shape [batch_size]

        index_tensor = activations_indices_last_token.long()

        # We need to create an additional dimension for the index tensor
        index_tensor = index_tensor.view(-1, 1, 1)  # Shape [batch_size, 1, 1]

        # Now we can gather along the second dimension
        activations_last_token = torch.gather(activation, 1, index_tensor.expand(-1, -1, activation.shape[-1])).squeeze(1)  # Shape [batch_size, hidden_size] (60, 1024)

        # Cut-off the maximum number of fmri features because of memory issues.
        test_data = test_data.view(-1, test_data.shape[-1])
        if max_fmri_features is not None:
            test_data = test_data[:, :max_fmri_features]

        # Split the data into train, validation and test.
        num_train_samples = int(train_perc * activations_last_token.shape[0])
        num_val_samples = activations_last_token.shape[0] - num_train_samples

        activations_last_token_train = activations_last_token[:num_train_samples]
        test_data_train = test_data[:num_train_samples]

        if num_val_samples < 2:
            print(f'Calculating the R^2 score requires at least 2 validation samples, but got {num_val_samples}. Using train for validation too.')
            activations_last_token_val = activations_last_token_train
            test_data_val = test_data_train
        else:
            activations_last_token_val = activations_last_token[num_train_samples:num_train_samples + num_val_samples]
            test_data_val = test_data[num_train_samples:num_train_samples + num_val_samples]

        if score_per_feature:
            brain_score_list = []
            for index in range(test_data.shape[1]):
                feature = test_data_train[:, index]
                clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(activations_last_token_train,
                                                                feature)
                feature_val = test_data_val[:, index]
                brain_score_list.append(clf.score(activations_last_token_val, feature_val))
                print(f'Brain score for feature {index}: {brain_score_list[-1]}')
            brain_scores['layer.module'].append(layer_name)
            brain_scores['brain_score'].append(sum(brain_score_list) / len(brain_score_list))  # Average
        else:
            clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(activations_last_token_train,
                                                            test_data_train)
            brain_scores['layer.module'].append(layer_name)
            brain_scores['brain_score'].append(clf.score(activations_last_token_val, test_data_val))

    return brain_scores


if __name__ == '__main__':
    # TODO: Not only taking the last token, but also the end of every fmri datapoint. You need to take the last token of every sentence.
    # TODO: Having score for multiple models. (on multiple checkpoints of the training)

    # Also pick the right difumo model.

    # Note, this is expected to be run from the root of the project. (not src).
    # In Pycharm, can click top-right left of the run button, expand dropdown menu, click on three dots next to calculate_brain_scores, click Edit
    # and set the working directory to the root of the project.
    # Specify parameters
    # path_to_model = r'models/cm_roberta-large.pt'  # Specify the path to the model.
    checkpoint_name = 'roberta-large'  # Specify the checkpoint name of the model. 'bert-base-cased' | 'roberta-large'

    # Load our custom pre-trained model on ETHICS and fMRI data.
    train_head_dims = [2, 39127]  # Need to fill this in to not get an error when loading the model. This is not used in the brain score calculation.
    from transformers import RobertaTokenizer, RobertaModel

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    model = RobertaModel.from_pretrained('roberta-large')
    # model = AutoModel.from_pretrained(checkpoint_name)
    # # model = BERT(model, head_dims=train_head_dims)
    # model.load_state_dict(torch.load(path_to_model))

    # Load Roberta model from local files.
    # model_config = AutoConfig.from_pretrained('roberta-large', num_labels=1)
    # model = RobertaModel.from_pretrained(path_to_model, local_files_only=True, config=model_config)
    # tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)

    # for name, layer in model.named_modules():
    #     print(name)

    # Load the data
    test_data_path = Path(environ.get('AISCBB_DATA_DIR','./data'))
    config = get_config()
    config['batch_size'] = 2  # Make the batch large enough so we definitely have one subject. This is a bit hacky but works for now.
    subjects = [f'sub-{i:02}' for i in range(3, 4)]  # TODO: there are missing subjects, so catch this here already.

    all_brain_scores = {'subjects': [], 'layer.module': [], 'brain_score': []}
    for subject in subjects:
        fmri_data = load_ds000212(test_data_path, tokenizer, config, subject=subject)
        data = iter(fmri_data[0]).next()  # Get the first batch of data which is one entire subject.
        model_inputs = (data[0], data[1])
        test_data = data[2]  # Shape (batch_size, num_features) (60, 1024) for a single participant.

        # Calculate the brain scores
        layers = ['23']   # The layers to calculate the brain scores for.
        modules = ['output.dense']  # The layer and module will be combined to 'layer.module' to get the activations.
        max_fmri_features = None  # This is used to limit the size of the data so that everything can still fit in memory. If None, all features are used.
        score_per_feature = True  # If True, a separate model is fitted for every feature. If False, a single model is fitted for all features.
        train_perc = 0.8  # The percentage of the data to use for training.
        val_perc = 0.2  # The percentage of the data to use for validation. If setting validation on 0, use the training data for validation too.
        brain_scores = calculate_brain_scores(model,
                                              model_inputs,
                                              test_data,
                                              layers, modules,
                                              max_fmri_features,
                                              score_per_feature=score_per_feature,
                                              train_perc=train_perc,
                                              val_perc=val_perc)

        # Add the brain scores to the all_brain_scores dict.
        all_brain_scores['subjects'].append(subject)
        all_brain_scores['layer.module'].extend(brain_scores['layer.module'])
        all_brain_scores['brain_score'].extend(brain_scores['brain_score'])

    print(all_brain_scores)

    # Write the brain scores to a csv file.
    path_to_brain_scores = os.path.join(os.getcwd(), 'artifacts', 'brain_scores')
    if not os.path.exists(path_to_brain_scores):
        os.makedirs(path_to_brain_scores)
    df = pd.DataFrame(all_brain_scores).to_csv(os.path.join(
        os.getcwd(), 'artifacts', 'brain_scores',
        f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'), index=False,
        sep=';')

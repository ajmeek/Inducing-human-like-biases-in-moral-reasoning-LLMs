import csv
import os
import yaml
from datetime import datetime
from os import environ
from pathlib import Path, WindowsPath
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
import numpy as np

from Context import Context
from utils.BrainBiasDataModule import BrainBiasDataModule
from pl_model import PLModel

from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr
from sklearn.feature_selection import r_regression


def return_path_to_latest_checkpoint() -> Path:
    """
    Iterates through the artifacts folder to find the latest saved checkpoint for calculation of brain scores
    :return: path to checkpoint
    """
    artifactspath = Path(environ.get('AISCBB_ARTIFACTS_DIR', '/artifacts'))
    subdirectories = [d for d in os.listdir(artifactspath) if os.path.isdir(os.path.join(artifactspath, d))]
    if not subdirectories:
        return None

    # sort subdirectories to get most recent directory first
    # technically, checkpoint names aren't fully accurate to your specific timezone necessarily. but they're consistent
    subdirectories_sorted = []
    for i in subdirectories:
        if i != 'brain_scores':
            converted = datetime.strptime(i, '%y%m%d-%H%M%S')
            subdirectories_sorted.append((converted, i))

    subdirectories_sorted_final = sorted(subdirectories_sorted, key=lambda x: x[0], reverse=True)

    for i in subdirectories_sorted_final:

        # checkpoints path
        checkpoint_path = Path(i[1] + '/version_0/checkpoints')
        checkpoint_path = os.path.join(artifactspath, checkpoint_path)

        if os.path.isdir(checkpoint_path):
            items_in_directory = os.listdir(checkpoint_path)
            if len(items_in_directory) == 1:
                # Get the path to the single item in the directory
                item_name = items_in_directory[0]
                item_path = os.path.join(checkpoint_path, item_name)
                return item_path
            else:
                # error, multiple checkpoints from one run.
                print("Error in return path to latest checkpoint, multiple checkpoints in run ", i[1])



def load_from_checkpoint(model, context):
    """
    This util function loads a lightning .ckpt checkpoint file into a HF model.
    Up to the user to make sure they're loading compatible models.
    For instance, load a deberta checkpoint into a deberta model and a bert base ckpt into a bert base model.

    :param model: the HF model to load into
    :param context: where to find the path, or if none, use return path util func above.
    :return: the HF model that's been loaded into (pass by reference and can eliminate? forgot how that works in Python)
    """

    # TODO - for some reason the lightning checkpoint has extra data for the weights and biases of the heads
    # check that the checkpoint is also the same as the bert base cased.
    # for now, ignore and continue. I did change replace("model." to "model.base." for future reference
    # done - just took them out.

    if context.finetuned_path is not None:
        path_to_model = Path(context.finetuned_path)

        state_dict = torch.load(path_to_model)
        state_dict_hf = state_dict['state_dict']
        state_dict_hf = {k.replace('model.base.', ''): state_dict_hf[k] for k in state_dict_hf}
        state_dict_hf.pop('model.heads.0.weight')
        state_dict_hf.pop('model.heads.0.bias')
        state_dict_hf.pop('model.heads.1.weight')
        state_dict_hf.pop('model.heads.1.bias')
        model.load_state_dict(state_dict_hf)
    else:
        path_to_model = return_path_to_latest_checkpoint()

        state_dict = torch.load(path_to_model)
        state_dict_hf = state_dict['state_dict']
        state_dict_hf = {k.replace('model.base.', ''): state_dict_hf[k] for k in state_dict_hf}
        state_dict_hf.pop('model.heads.0.weight')
        state_dict_hf.pop('model.heads.0.bias')
        state_dict_hf.pop('model.heads.1.weight')
        state_dict_hf.pop('model.heads.1.bias')
        model.load_state_dict(state_dict_hf)
    return model


def calculate_brain_scores(model: nn.Module,
                           model_inputs: torch.tensor,
                           test_data: torch.tensor,
                           layer: list,
                           modules: list,
                           finetuned: bool,
                           train_perc: float = 1.0,
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
    :param train_perc: The percentage of the data to use for training.
    :param val_perc: The percentage of the data to use for validation.
    :param finetuned: Bool indicating whether or not the model is finetuned. For print out
    :return: A dictionary containing the brain scores for each layer and module.
    """
    activations = {}
    all_layer_names = []
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
    """
    layer.module = The layer that the brain score is calculated on
    brain_score = The overall brain score. Pearson's correlation coefficient of ground truth fMRI ROI activations and
        the ridge regression model's predictions for each fMRI ROI
    coeff_of_det = The coefficient of determination for each fMRI ROI. 
    ridge_regress_predict = The prediction of the ridge regression model for each fMRI ROI.
    """
    brain_scores = {'layer.module': [], 'brain_score': [],
                    'coeff_of_det': [], 'ridge_regress_predict': []}
    for layer_name, activation in activations.items():
        print('Calculating brain score for layer:', layer_name,
              ' activation dims: ', activation.shape,
              ' finetuned: ', finetuned)

        # activations_flattened = activation.reshape(activation.shape[0], -1)
        activations_indices_last_token = torch.sum(attention_mask, dim=1) - 1  # Shape [batch_size]

        index_tensor = activations_indices_last_token.long()

        # We need to create an additional dimension for the index tensor
        index_tensor = index_tensor.view(-1, 1, 1)  # Shape [batch_size, 1, 1]

        # Now we can gather along the second dimension
        activations_last_token = torch.gather(activation, 1, index_tensor.expand(-1, -1, activation.shape[-1])).squeeze(
            1)  # Shape [batch_size, hidden_size] (60, 1024)

        # Split the data into train, validation and test.
        num_train_samples = int(train_perc * activations_last_token.shape[0])
        num_val_samples = activations_last_token.shape[0] - num_train_samples

        activations_last_token_train = activations_last_token[:num_train_samples]
        test_data_train = test_data[:num_train_samples]

        if num_val_samples < 2:  # and correlation == "determination":
            print(
                f'Calculating the R^2 score requires at least 2 validation samples, but got {num_val_samples}. Using train for validation too.')
            activations_last_token_val = activations_last_token_train
            test_data_val = test_data_train
        else:
            activations_last_token_val = activations_last_token[num_train_samples:num_train_samples + num_val_samples]
            test_data_val = test_data[num_train_samples:num_train_samples + num_val_samples]

        brain_score_list = []

        # Coefficient of determination
        for index in range(test_data.shape[1]):
            feature = test_data_train[:, index]
            clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(activations_last_token_train,
                                                            feature)
            feature_val = test_data_val[:, index]
            brain_score_list.append(clf.score(activations_last_token_val, feature_val))
            brain_scores['coeff_of_det'].append(brain_score_list[-1])

        # Pearson's correlation coefficient and ridge regression .predict() for each ROI
        activations_last_token_train = activations_last_token[:-1]
        test_data_train = test_data[:-1]
        activations_last_token_val = activations_last_token[-1:]
        test_data_val = test_data[-1:]

        predictions_list = []
        for index in range(test_data.shape[1]):
            feature = test_data_train[:, index]
            clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(activations_last_token_train,
                                                            feature)
            prediction = clf.predict(activations_last_token_val)

            brain_scores['ridge_regress_predict'].append(prediction[0])
            predictions_list.append(prediction)

        # reshape test data val to be 1d of shape (1024), not (1, 1024)
        test_data_val = test_data_val[0, :]
        pearson_r = r_regression(predictions_list, test_data_val)

        brain_scores['layer.module'].append(layer_name)
        # brain_scores['brain_score'].append(sum(brain_score_list) / len(range(test_data.shape[1])))  # Average
        brain_scores['brain_score'].append(pearson_r[0])  # Pearson's r

    return brain_scores


if __name__ == '__main__':
    # Note, this is expected to be run from the root of the project. (not src).
    # In Pycharm, can click top-right left of the run button, expand dropdown menu, click on three dots next to calculate_brain_scores, click Edit
    # and set the working directory to the root of the project.

    context = Context()

    # rewriting and getting rid of the wrapper. change above brain score function to accept one layer at a time
    layers = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    subject_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 27, 28, 29, 30, 31,
                    32, 33, 34, 35, 38, 39, 40, 41, 42, 44, 45, 46, 47]
    subjects = [f'sub-{i:02}' for i in subject_list]
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") #for naming
    train_perc = 0.8 #for ridge regression model

    # base model - not finetuned
    model = AutoModel.from_pretrained(context.model_path)
    tokenizer = AutoTokenizer.from_pretrained(context.model_path)
    data_module = BrainBiasDataModule(context.get_ds_configs(), tokenizer)
    base_model = PLModel(model, context.plc, data_module)


    # finetuned model
    finetuned_model = load_from_checkpoint(model, context)
    data_module = BrainBiasDataModule(context.get_ds_configs(), tokenizer)
    finetuned_model = PLModel(finetuned_model, context.plc, data_module)

    # dataset
    ds = load_dataset('/home/austin/PycharmProjects/Inducing-human-like-biases-in-moral-reasoning-LLMs/data/ds000212/ds000212_lfb', name='LFB-LAST')

    # for writing to file. Keep same data structure as before so Seong's notebooks don't break
    # actually Seong, it may be easiest for me to
    all_brain_scores = {'subjects': [], 'layer.module': [], 'brain_score': []}
    coeff_of_det_per_subject = {}
    ridge_regress_predict_per_subject = {}

    for finetuned in [False, True]:

        path_to_brain_scores = os.path.join(os.getcwd(), 'artifacts', 'brain_scores',
                                            f'{date}_finetuned={finetuned}')  # change finetune for below loop
        if not os.path.exists(path_to_brain_scores):
            os.makedirs(path_to_brain_scores)

        for subject in subjects:

            # get data for each subject from the dataloader, from the train split
            subject_data = list(ds.filter(lambda e: subject in e['file'])['train'])

            """
            Brain scores expects three main things:
                - fmri data - in shape [batch_size, length=1024]
                - tokens - tokens from input, in shape [batch_size, sequence length]
                - attention mask - similarly to the above. This and the above are passed as the tuple model_inputs.
    
            Since I now have the above way to load in per-subject data instead of using the dataloader with set batch size,
            pass as big of a batch as possible so the ridge regression can better fit to the model.
            """

            fmri_data = torch.tensor([e['label'] for e in subject_data])
            inputs = [e['input'] for e in subject_data]
            tokenized = tokenizer(inputs, padding='max_length', truncation=True)
            tokens = torch.tensor(tokenized['input_ids'])
            attention_mask = torch.tensor(tokenized['attention_mask'])
            model_inputs = (tokens, attention_mask)

            path_to_brain_scores_subj = os.path.join(path_to_brain_scores, subject)
            if not os.path.exists(path_to_brain_scores_subj):
                os.makedirs(path_to_brain_scores_subj)

            for layer in layers:
                modules = ['output.dense'] #combined with layer to get the activations

                if finetuned:
                    brain_scores = calculate_brain_scores(finetuned_model, model_inputs, fmri_data, layer, modules, train_perc=train_perc, finetuned=finetuned)
                elif not finetuned:
                    brain_scores = calculate_brain_scores(base_model, model_inputs, fmri_data, layer, modules, train_perc=train_perc, finetuned=finetuned)

                # Add the brain scores to the all_brain_scores dict.
                all_brain_scores['subjects'].append(subject)
                all_brain_scores['layer.module'].extend(brain_scores['layer.module'])
                all_brain_scores['brain_score'].extend(brain_scores['brain_score'])
                coeff_of_det_per_subject[subject] = brain_scores['coeff_of_det']
                ridge_regress_predict_per_subject[subject] = brain_scores['ridge_regress_predict']

                df = pd.DataFrame((brain_scores['coeff_of_det'], brain_scores['ridge_regress_predict'])).to_csv(
                    os.path.join(path_to_brain_scores_subj, f'layer_{layer}_brain_scores_per_feature.csv'),
                    index=False, sep=',')


            # Create a text file and save it to path_to_brain_scores with metadata
            path_to_metadata_file = path_to_brain_scores_subj + '/metadata.txt'
            with open(path_to_metadata_file, 'w') as f:
                f.write(f"Model: {context.model_path}\n") #remember, user should confirm that finetuned model same as base model
                f.write(f"Layers: {layers}\n")
                f.write(f"Finetuned: False\n")
                f.write(f"Train percentage: {train_perc}\n")
                f.write(f"Date: {date}\n")
                f.write("\n Checkpoint metadata below (if finetuned): \n")

                if finetuned:
                    # load metadata from checkpoint as well - built off of lightning's checkpoint directory structure
                    path_to_model = return_path_to_latest_checkpoint()
                    parent_path = os.path.dirname(path_to_model)
                    grandparent_path = os.path.dirname(parent_path)

                    yaml_path = os.path.join(grandparent_path, 'hparams.yaml')
                    with open(yaml_path, 'r') as file:
                        yaml_contents = yaml.load(file, Loader=yaml.FullLoader)

                        for i in yaml_contents.keys():
                            f.write(f"{i}: {yaml_contents[i]}\n")

            break #just for one subj to test
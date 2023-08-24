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
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig, RobertaModel

from src.main import get_config
from src.model import BERT
#from src.utils.loading_data import load_ds000212
from src.utils.loading_data import return_path_to_latest_checkpoint
from main import get_config
from model import BERT
from utils.DS000212_LFB_Dataset import DS000212_LFB_Dataset

from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr
from sklearn.feature_selection import r_regression

from utils.constants import Sampling

def calculate_brain_scores(model: nn.Module,
                           model_inputs: torch.tensor,
                           test_data: torch.tensor,
                           layers: list,
                           modules: list,
                           max_fmri_features: Optional[int],
                           finetuned: bool,
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
    :param finetuned: Bool indicating whether or not the model is finetuned. For print out
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
    """
    layer.module = The layer that the brain score is calculated on
    brain_score = The overall brain score. Pearson's correlation coefficient of ground truth fMRI ROI activations and
        the ridge regression model's predictions for each fMRI ROI
    brain_score_positive = deprecated. Was a sum of only non-negative values from coeff_of_det
    brain_score_per_feature = deprecated. Was score for each feature, dependent on choice of R^2 or pearson's r
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
        activations_last_token = torch.gather(activation, 1, index_tensor.expand(-1, -1, activation.shape[-1])).squeeze(1)  # Shape [batch_size, hidden_size] (60, 1024)

        # Cut-off the maximum number of fmri features because of memory issues.
        test_data = test_data.view(-1, test_data.shape[-1])
        if max_fmri_features is not None:
            test_data = test_data[:, :max_fmri_features]

        # Split the data into train, validation and test.
        num_train_samples = int(train_perc * activations_last_token.shape[0])
        num_val_samples = activations_last_token.shape[0] - num_train_samples

        # TODO - should the validation activation data, used for ROI brain score, be a specific part of the scenario?
        # scenarios are split into different sections. Perhaps some sections are more morally relevant than others?

        #note, we could use all but one for pearson's but need at least 2 for coeff of det. Doing them together means to
        #train two separate models (minor computational expense) if we want 1 extra data point in pearson's train set.
        #With Artyom's new sampling method, this will likely not be an issue.
        # if correlation == "pearson":
        #     num_train_samples = 7
        #     num_val_samples = 1

        activations_last_token_train = activations_last_token[:num_train_samples]
        test_data_train = test_data[:num_train_samples]

        if num_val_samples < 2:# and correlation == "determination":
            print(f'Calculating the R^2 score requires at least 2 validation samples, but got {num_val_samples}. Using train for validation too.')
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

            # TODO - compare on finetuned on ethics v pretrained, and finetued on fMRI v pretrained (separate in train CLI command somehow)

            # TODO - send the predictions above for ROI scores, along with R^2 scores

            # TODO - integrate Artyom's new sampling, fit on more data. predict on maximally moral reasoning TR

            brain_scores['ridge_regress_predict'].append(prediction[0])
            predictions_list.append(prediction)

        #reshape test data val to be 1d of shape (1024), not (1, 1024)
        test_data_val = test_data_val[0,:]
        pearson_r = r_regression(predictions_list, test_data_val)


        brain_scores['layer.module'].append(layer_name)
        #brain_scores['brain_score'].append(sum(brain_score_list) / len(range(test_data.shape[1])))  # Average
        brain_scores['brain_score'].append(pearson_r[0]) # Pearson's r

        # else:
        #     clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(activations_last_token_train,
        #                                                     test_data_train)
        #     brain_scores['layer.module'].append(layer_name)
        #     brain_scores['brain_score'].append(clf.score(activations_last_token_val, test_data_val))

    return brain_scores


if __name__ == '__main__':
    # Note, this is expected to be run from the root of the project. (not src).
    # In Pycharm, can click top-right left of the run button, expand dropdown menu, click on three dots next to calculate_brain_scores, click Edit
    # and set the working directory to the root of the project.
    # Specify parameters
    # path_to_model = r'models/cm_roberta-large.pt'  # Specify the path to the model.


    def wrapper(path_to_model, layer_list, date, finetuned):
        """
        This wrapper abstracts the running of the code to loop over all possibilities.

        Params:
        path_to_model = this is the path to the most recent checkpoint of the finetuned model. N/A when training on base BERT
        layer_list = list of layers you want to find brain scores for
        finetuned = Boolean. whether or not to test on the finetuned model.
        """
        checkpoint_name = 'bert-base-cased'  # Specify the checkpoint name of the model. 'bert-base-cased' | 'roberta-large'

        # Load our custom pre-trained model on ETHICS and fMRI data.

        # Warning - training head dims should match difumo resolution
        train_head_dims = [2, 1024]
        model = AutoModel.from_pretrained(checkpoint_name)
        model = BERT(model, head_dims=train_head_dims)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)

        if finetuned:
            # manually adjusting state dict so that lightning models fit with HF
            # there are 8 dictionary entries specific to lightning that are not needed. everything else should be the same
            state_dict = torch.load(path_to_model)
            state_dict_hf = state_dict['state_dict']

            state_dict_hf = {k.replace('model.', ''): state_dict_hf[k] for k in state_dict_hf}

            model.load_state_dict(state_dict_hf)

        # Load the data
        context = get_config()
        context['datapath'] = Path(environ.get('AISCBB_DATA_DIR','./data'))
        #context['batch_size'] = 2  # Make the batch large enough so we definitely have one subject. This is a bit hacky but works for now.
        context['batch_size'] = 2
        context['sampling_method'] = Sampling.SENTENCES
        #subjects = [f'sub-{i:02}' for i in range(3, 4)]
        subject_list = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,24,27,28,29,30,31,32,33,34,35,38,39,40,41,42,44,45,46,47]
        subjects = [f'sub-{i:02}' for i in subject_list]


        all_brain_scores = {'subjects': [], 'layer.module': [], 'brain_score': []}

        features_per_subject = {}
        coeff_of_det_per_subject = {}
        ridge_regress_predict_per_subject = {}
        # for subject in subjects:
        #     fmri_data = load_ds000212(test_data_path, tokenizer, config, subject=subject, intervals=[2, 4, 6, 8])  # Use [2, 4, 6, 8] to use the background, action, outcome, and skind. Use -1 to use only the last fMRI.
        #     data = next(iter(fmri_data[0]))  # Get the first batch of data which is one entire subject.
        #     model_inputs = (data[0], data[1])
        #     test_data = data[2]  # Shape (batch_size, num_features) (60, 1024) for a single participant.
        all_brain_scores = {'subjects': [], 'layer.module': [], 'brain_score': []}
        for subject in subjects:
            ds000212 = DS000212_LFB_Dataset(context, tokenizer, subject=subject)
            fmri_data = DataLoader(
                ds000212,
                batch_size=context['batch_size']
            )
            #correct_time_points = ds000212.sample_from_bold_sequence(fmri_data, Sampling.SENTENCES)
            data = next(iter(fmri_data)) # Get the first batch of data which is one entire subject.
            model_inputs = (data[0], data[1])
            test_data = data[2]  # Shape (batch_size, num_features) (60, 1024) for a single participant.

            print("test_data.shape: ", test_data.shape)
            print("test_data: ", test_data)
            break
            #correct_time_points = ds000212.sample_from_bold_sequence(test_data, Sampling.SENTENCES)

            # Calculate the brain scores
            layers = layer_list   # The layers to calculate the brain scores for.
            modules = ['output.dense']  # The layer and module will be combined to 'layer.module' to get the activations.
            max_fmri_features = None  # This is used to limit the size of the data so that everything can still fit in memory. If None, all features are used.
            score_per_feature = True  # If True, score per feature output. If False, not output but still calculated for Pearson's correlation coefficient.
            train_perc = 0.8  # The percentage of the data to use for training.
            val_perc = 0.2  # The percentage of the data to use for validation. If setting validation on 0, use the training data for validation too.

            print(f"Running brain scores on subject {subject} and finetuned is {finetuned}")
            brain_scores = calculate_brain_scores(model,
                                                  model_inputs,
                                                  test_data,
                                                  layers, modules,
                                                  max_fmri_features,
                                                  #correlation=correlation,
                                                  score_per_feature=score_per_feature,
                                                  train_perc=train_perc,
                                                  val_perc=val_perc,
                                                  finetuned=finetuned)

            # Add the brain scores to the all_brain_scores dict.
            all_brain_scores['subjects'].append(subject)
            all_brain_scores['layer.module'].extend(brain_scores['layer.module'])
            all_brain_scores['brain_score'].extend(brain_scores['brain_score'])
            coeff_of_det_per_subject[subject] = brain_scores['coeff_of_det']
            ridge_regress_predict_per_subject[subject] = brain_scores['ridge_regress_predict']

        print('subjects: ', all_brain_scores['subjects'])
        print('layers: ', all_brain_scores['layer.module'])
        print('brain_score: ', all_brain_scores['brain_score'])

        # Write the brain scores to a csv file.
        # TODO - move inside the main folder
        path_to_brain_scores = os.path.join(os.getcwd(), 'artifacts', 'brain_scores', f'{date}_layer={layer_list[0]}_finetuned={finetuned}')
        if not os.path.exists(path_to_brain_scores):
            os.makedirs(path_to_brain_scores)
        df = pd.DataFrame(all_brain_scores).to_csv(os.path.join(
            os.getcwd(), 'artifacts', 'brain_scores',
            f'{date}_layer={layer_list[0]}_finetuned={finetuned}',
            f'{date}_finetuned={finetuned}.csv'), index=False,
            sep=',')

        #Create a text file and save it to path_to_brain_scores with metadata
        path_to_metadata_file = path_to_brain_scores + '/metadata.txt'
        with open(path_to_metadata_file, 'w') as f:
            f.write(f"Model: {checkpoint_name}\n")
            f.write(f"Layers: {layer_list}\n")
            f.write(f"Finetuned: {finetuned}\n")
            f.write(f"Score per feature: {score_per_feature}\n")
            f.write(f"Train percentage: {train_perc}\n")
            f.write(f"Validation percentage: {val_perc}\n")
            f.write(f"Date: {date}\n")
            f.write("\n Checkpoint metadata below: \n")

            #load metadata from checkpoint as well
            parent_path = os.path.dirname(path_to_model)
            grandparent_path = os.path.dirname(parent_path)

            yaml_path = os.path.join(grandparent_path, 'hparams.yaml')
            with open(yaml_path, 'r') as file:
                yaml_contents = yaml.load(file, Loader=yaml.FullLoader)

                for i in yaml_contents.keys():
                    f.write(f"{i}: {yaml_contents[i]}\n")

        if score_per_feature:
            for i in coeff_of_det_per_subject.keys():

                # TODO - have a text file in the folder that is output to with metadata, such as the checkpoint metadata

                df = pd.DataFrame((coeff_of_det_per_subject[i], ridge_regress_predict_per_subject[i])).to_csv(os.path.join(
                    path_to_brain_scores, f'subject_{i}_brain_scores_per_feature.csv'),
                    index=False, sep=',')
        #print("coeff of det per subject: ", coeff_of_det_per_subject, "\nridge regress per subject: ", ridge_regress_predict_per_subject)

    #base BERT has 12 encoder layers
    layer_list = ['2','3','4','5','6','7','8','9','10','11']#,'12'] #including layer 1 and 12 breaks it for some reason
    path_to_model = return_path_to_latest_checkpoint()
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") #for naming
    #wrapper(path_to_model, layer_list, finetuned=False)

    #instead of using layer list, just pass a single layer to the wrapper function and loop using that.
    #will need to incorporate the layer into the directory name then.
    for i in layer_list:
        wrapper(path_to_model, [i], date, finetuned=False)
        wrapper(path_to_model, [i], date, finetuned=True)
        #break

    #wrapper(path_to_model, ['2'], date, finetuned=False)
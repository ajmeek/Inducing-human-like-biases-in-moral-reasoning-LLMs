import csv
import os
from datetime import datetime
from os import environ
from pathlib import Path, WindowsPath
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig, RobertaModel

from src.main import get_config
from src.model import BERT
from src.utils.loading_data import load_ds000212
from src.utils.loading_data import return_path_to_latest_checkpoint
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr
from sklearn.feature_selection import r_regression


def calculate_brain_scores(model: nn.Module,
                           model_inputs: torch.tensor,
                           test_data: torch.tensor,
                           layers: list,
                           modules: list,
                           max_fmri_features: Optional[int],
                           correlation: str,
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
    :param correlation: Choose either Pearson's r as "pearson" or the coefficient of determination as "determination"
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
    brain_scores = {'layer.module': [], 'brain_score': [], 'brain_score_positive': [], 'brain_score_per_feature': [], 'correlation': []}
    for layer_name, activation in activations.items():
        print('Calculating brain score for layer:', layer_name,
              ' activation dims: ', activation.shape, '...',
              ' correlation: ', correlation)
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

        if correlation == "pearson":
            num_train_samples = 7
            num_val_samples = 1

        activations_last_token_train = activations_last_token[:num_train_samples]
        test_data_train = test_data[:num_train_samples]

        if num_val_samples < 2 and correlation == "determination":
            print(f'Calculating the R^2 score requires at least 2 validation samples, but got {num_val_samples}. Using train for validation too.')
            activations_last_token_val = activations_last_token_train
            test_data_val = test_data_train
        else:
            activations_last_token_val = activations_last_token[num_train_samples:num_train_samples + num_val_samples]
            test_data_val = test_data[num_train_samples:num_train_samples + num_val_samples]

        if score_per_feature:
            brain_score_list = []
            if correlation == "determination":
                for index in range(test_data.shape[1]):
                    feature = test_data_train[:, index]
                    clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(activations_last_token_train,
                                                                    feature)
                    feature_val = test_data_val[:, index]
                    brain_score_list.append(clf.score(activations_last_token_val, feature_val))
                    #print(f'Brain score for feature {index}: {brain_score_list[-1]}')
                    brain_scores['brain_score_per_feature'].append((index, brain_score_list[-1]))
            elif correlation == "pearson":
                predictions_list = []
                for index in range(test_data.shape[1]):
                    #feature = test_data[:, index]
                    feature = test_data_train[:, index]
                    clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(activations_last_token_train,
                                                                    feature)
                    feature_val = test_data_val[:, index]
                    prediction = clf.predict(activations_last_token_val)
#                    pearson_r = r_regression(predictions, feature_val)

                    # TODO - does Seong want just the prediction or the correlation between prediction and actual?

                    predictions_list.append(prediction)

                    #brain_score_list.append(clf.score(activations_last_token_val, feature_val))
                    #pearson_r_train, _ = pearsonr(activations_last_token_train, feature)
                    #pearson_r_train = r_regression(activations_last_token_train, feature)
                    #feature_val = test_data_val[:, index]
                    #pearson_r_val, _ = pearsonr(activations_last_token_val, feature_val)
                    #pearson_r_val = r_regression(activations_last_token_val, feature_val)
                    #pearson_r = r_regression(activations_last_token, feature)
                    #brain_score_list.append(pearson_r)
                    #print(f'Brain score for feature {index}: {brain_score_list[-1]}')

                #reshape test data val to be 1d of shape (1024), not (1, 1024)
                test_data_val = test_data_val[0,:]
                predictions_list_flipped = np.array(predictions_list)
                #predictions_list_flipped =
                pearson_r = r_regression(predictions_list, test_data_val)

                # TODO - loop over every index again.
                """
                How this works is that Pearson's calculates a pairwise thing, but it's out of many as I understand it.
                So how much does one predicted ROI match the actual ROI activation, in comparison to all the other predicted ROI data
                
                So need to loop over every index again and calculate it out. Just need to make sure that my indices are right.
                
                According to docs, r_regression(X,Y) = X = (num_samples, num_features) and Y = (num_samples,).
                We have one sample (one subj, layer, model combo) and 1024 ROI features. So flip predictions_list to be
                (1, 1024).
                """

                brain_scores['brain_score_per_feature'] = brain_score_list

            brain_scores['layer.module'].append(layer_name)
            brain_scores['brain_score'].append(sum(brain_score_list) / len(brain_score_list))  # Average
            brain_scores['correlation'].append(correlation)

            brain_score_positive = 0
            for i in brain_score_list:
                if i >= 0:
                    brain_score_positive += i
            brain_scores['brain_score_positive'].append(brain_score_positive)
        else:
            clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(activations_last_token_train,
                                                            test_data_train)
            brain_scores['layer.module'].append(layer_name)
            brain_scores['brain_score'].append(clf.score(activations_last_token_val, test_data_val))

    return brain_scores


if __name__ == '__main__':
    # Note, this is expected to be run from the root of the project. (not src).
    # In Pycharm, can click top-right left of the run button, expand dropdown menu, click on three dots next to calculate_brain_scores, click Edit
    # and set the working directory to the root of the project.
    # Specify parameters
    # path_to_model = r'models/cm_roberta-large.pt'  # Specify the path to the model.


    def wrapper(path_to_model, layer_list, date, correlation, finetuned):
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
        test_data_path = Path(environ.get('AISCBB_DATA_DIR'))
        config = get_config()
        config['batch_size'] = 2  # Make the batch large enough so we definitely have one subject. This is a bit hacky but works for now.
        subjects = [f'sub-{i:02}' for i in range(3, 4)]
        subject_list = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,24,27,28,29,30,31,32,33,34,35,38,39,40,41,42,44,45,46,47]
        subjects = [f'sub-{i:02}' for i in subject_list]

        all_brain_scores = {'subjects': [], 'layer.module': [], 'brain_score': [], 'brain_score_positive': [], 'correlation': []}#, 'brain_score_per_feature': []}

        features_per_subject = {}
        for subject in subjects:
            fmri_data = load_ds000212(test_data_path, tokenizer, config, subject=subject, intervals=[2, 4, 6, 8])  # Use [2, 4, 6, 8] to use the background, action, outcome, and skind. Use -1 to use only the last fMRI.
            data = next(iter(fmri_data[0]))  # Get the first batch of data which is one entire subject.
            model_inputs = (data[0], data[1])
            test_data = data[2]  # Shape (batch_size, num_features) (60, 1024) for a single participant.

            # Calculate the brain scores
            layers = layer_list   # The layers to calculate the brain scores for.
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
                                                  correlation=correlation,
                                                  score_per_feature=score_per_feature,
                                                  train_perc=train_perc,
                                                  val_perc=val_perc)

            # Add the brain scores to the all_brain_scores dict.
            all_brain_scores['subjects'].append(subject)
            all_brain_scores['layer.module'].extend(brain_scores['layer.module'])
            all_brain_scores['brain_score'].extend(brain_scores['brain_score'])
            all_brain_scores['brain_score_positive'].extend(brain_scores['brain_score_positive'])
            all_brain_scores['correlation'].extend(brain_scores['correlation'])
            #all_brain_scores['brain_score_per_feature'] = brain_scores['brain_score_per_feature']
            features_per_subject[subject] = brain_scores['brain_score_per_feature']

        print('subjects: ', all_brain_scores['subjects'])
        print('layers: ', all_brain_scores['layer.module'])
        print('brain_score: ', all_brain_scores['brain_score'])

        # Write the brain scores to a csv file.
        path_to_brain_scores = os.path.join(os.getcwd(), 'artifacts', 'brain_scores')
        if not os.path.exists(path_to_brain_scores):
            os.makedirs(path_to_brain_scores)
        df = pd.DataFrame(all_brain_scores).to_csv(os.path.join(
            os.getcwd(), 'artifacts', 'brain_scores',
            f'{date}.csv'), index=False,
            sep=',')

        #date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path_to_brain_scores = os.path.join(os.getcwd(), 'artifacts', 'brain_scores', f'{date}_layer={layer_list[0]}_finetuned={finetuned}_correlation={correlation}')
        if not os.path.exists(path_to_brain_scores):
            os.makedirs(path_to_brain_scores)
        for i in features_per_subject.keys():


            #print(features_per_subject[i])

            df = pd.DataFrame(features_per_subject[i]).to_csv(os.path.join(
                path_to_brain_scores, f'subject_{i}_brain_scores_per_feature.csv'),
                index=False, sep=',')

        """
        Thoughts - to write to csv like the above, all arrays need to be the same length.
        
        For now, have the csv write as normal like it was, and then have an additional folder that holds
        brain scores per feature for different subjects and layers. Going to eat lunch first though.
        
        TODO - Have this iterate over everything in the layer list as well.
        """

    #base BERT has 12 encoder layers
    layer_list = ['2','3','4','5','6','7','8','9','10','11','12'] #including layer 1 and 12 breaks it for some reason
    path_to_model = return_path_to_latest_checkpoint()
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") #for naming
    #wrapper(path_to_model, layer_list, finetuned=False)

    #instead of using layer list, just pass a single layer to the wrapper function and loop using that.
    #will need to incorporate the layer into the directory name then. 
    for i in layer_list:
        #print([i])
        wrapper(path_to_model, [i], date, correlation="pearson", finetuned=False)
        #wrapper(path_to_model, [i], date, correlation="pearson", finetuned=True)
        #wrapper(path_to_model, [i], date, correlation="determination", finetuned=False)
        #wrapper(path_to_model, [i], date, correlation="determination", finetuned=True)
        break

    #wrapper(path_to_model, ['2'], date, finetuned=False)
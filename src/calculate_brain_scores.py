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
from torch.utils.data import TensorDataset, DataLoader

from Context import Context
from utils.BrainBiasDataModule import BrainBiasDataModule
from pl_model import PLModel
from simple_parsing import ArgumentParser


# commented out for now while I rework the brain score functionality
# from src.main import get_config
# from src.model import BERT
# #from src.utils.loading_data import load_ds000212
# from src.utils.loading_data import return_path_to_latest_checkpoint
# from main import get_config
# from utils.DS000212_LFB_Dataset import DS000212_LFB_Dataset
#
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr
from sklearn.feature_selection import r_regression
#
# from utils.constants import Sampling

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


# path = return_path_to_latest_checkpoint()
# print(path)
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

    if context.finetuned_path is not None:
        path_to_model = Path(context.finetuned_path)

        state_dict = torch.load(path_to_model)
        state_dict_hf = state_dict['state_dict']
        state_dict_hf = {k.replace('model.base.', ''): state_dict_hf[k] for k in state_dict_hf}
        model.load_state_dict(state_dict_hf)
    else:
        path_to_model = return_path_to_latest_checkpoint()

        state_dict = torch.load(path_to_model)
        state_dict_hf = state_dict['state_dict']
        state_dict_hf = {k.replace('model.base.', ''): state_dict_hf[k] for k in state_dict_hf}
        model.load_state_dict(state_dict_hf)
    return model


def calculate_brain_scores(model: nn.Module,
                           model_inputs: torch.tensor,
                           test_data: torch.tensor,
                           layers: list,
                           modules: list,
                           max_fmri_features: Optional[int],
                           finetuned: bool,
                           score_per_feature: bool = False,
                           train_perc: float = 1.0,
                           val_perc: float = 0.0,
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
        activations_last_token = torch.gather(activation, 1, index_tensor.expand(-1, -1, activation.shape[-1])).squeeze(
            1)  # Shape [batch_size, hidden_size] (60, 1024)

        # Cut-off the maximum number of fmri features because of memory issues.
        test_data = test_data.view(-1, test_data.shape[-1])
        if max_fmri_features is not None:
            test_data = test_data[:, :max_fmri_features]

        # Split the data into train, validation and test.
        num_train_samples = int(train_perc * activations_last_token.shape[0])
        num_val_samples = activations_last_token.shape[0] - num_train_samples

        # TODO - should the validation activation data, used for ROI brain score, be a specific part of the scenario?
        # scenarios are split into different sections. Perhaps some sections are more morally relevant than others?

        # note, we could use all but one for pearson's but need at least 2 for coeff of det. Doing them together means to
        # train two separate models (minor computational expense) if we want 1 extra data point in pearson's train set.
        # With Artyom's new sampling method, this will likely not be an issue.
        # if correlation == "pearson":
        #     num_train_samples = 7
        #     num_val_samples = 1

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

            # TODO - compare on finetuned on ethics v pretrained, and finetued on fMRI v pretrained (separate in train CLI command somehow)

            # TODO - send the predictions above for ROI scores, along with R^2 scores

            # TODO - integrate Artyom's new sampling, fit on more data. predict on maximally moral reasoning TR

            brain_scores['ridge_regress_predict'].append(prediction[0])
            predictions_list.append(prediction)

        # reshape test data val to be 1d of shape (1024), not (1, 1024)
        test_data_val = test_data_val[0, :]
        pearson_r = r_regression(predictions_list, test_data_val)

        brain_scores['layer.module'].append(layer_name)
        # brain_scores['brain_score'].append(sum(brain_score_list) / len(range(test_data.shape[1])))  # Average
        brain_scores['brain_score'].append(pearson_r[0])  # Pearson's r

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

    context = Context()

    def wrapper(path_to_model, tokenizer, layer_list, date, finetuned):
        """
        This wrapper abstracts the running of the code to loop over all possibilities.

        Params:
        path_to_model = this is the path to the most recent checkpoint of the finetuned model. N/A when training on base BERT
        layer_list = list of layers you want to find brain scores for
        finetuned = Boolean. whether or not to test on the finetuned model.
        """
        checkpoint_name = 'bert-base-cased'  # Specify the checkpoint name of the model. 'bert-base-cased' | 'roberta-large'

        # Load our custom pre-trained model on ETHICS and fMRI data.

        # Note 13/9 why do the train head dimensions have 2,1024 instead of just 1024?
        # want to switch this to using the context structure that Artyom defined.

        # not sure why still. Also need to figure out how to replicate the old functionality we had with the BERT class
        # in src.model which Artyom replaced. Will look into it but if I can't figure it out I'll shoot him a message.

        # check the pl_model file. Artyom has automated the heads and their linear mapping to the correct dimensions.
        # alright, have this create a pl_model as in main.py and then check what I can get from the ._heads attribute,
        # which should have what I want.
        # No need to reinitialize the model over and over in this wrapper either. Let me move it below.

        # Load the data
        # context = get_config()
        # context['datapath'] = Path(environ.get('AISCBB_DATA_DIR', './data'))
        # # context['batch_size'] = 2  # Make the batch large enough so we definitely have one subject. This is a bit hacky but works for now.
        # context['batch_size'] = 2
        # context['sampling_method'] = Sampling.SENTENCES
        # subjects = [f'sub-{i:02}' for i in range(3, 4)]








        subject_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 27, 28, 29, 30, 31,
                        32, 33, 34, 35, 38, 39, 40, 41, 42, 44, 45, 46, 47]
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
            # ds000212 = DS000212_LFB_Dataset(context, tokenizer, subject=subject)
            # fmri_data = DataLoader(
            #     ds000212,
            #     batch_size=context['batch_size']
            # )
            # # correct_time_points = ds000212.sample_from_bold_sequence(fmri_data, Sampling.SENTENCES)
            # data = next(iter(fmri_data))  # Get the first batch of data which is one entire subject.
            # model_inputs = (data[0], data[1])
            # test_data = data[2]  # Shape (batch_size, num_features) (60, 1024) for a single participant.

            # TODO - How to get the attention mask for a given input?
            # alright, I'll need to tokenize it myself now. I should pass the tokenizer in along with the model to do so

            tokenized = tokenizer(inputs, padding='max_length', truncation=True)

            # convert tokens, masks, and targets into tensors
            tokens = torch.tensor(tokenized['input_ids'])
            masks = torch.tensor(tokenized['attention_mask'])
            model_inputs = TensorDataset(tokens, masks)

            # print("test_data.shape: ", test_data.shape)
            # print("test_data: ", test_data)
            # break
            # correct_time_points = ds000212.sample_from_bold_sequence(test_data, Sampling.SENTENCES)

            # Calculate the brain scores
            layers = layer_list  # The layers to calculate the brain scores for.
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
                                                  # correlation=correlation,
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
        path_to_brain_scores = os.path.join(os.getcwd(), 'artifacts', 'brain_scores',
                                            f'{date}_layer={layer_list[0]}_finetuned={finetuned}')
        if not os.path.exists(path_to_brain_scores):
            os.makedirs(path_to_brain_scores)
        df = pd.DataFrame(all_brain_scores).to_csv(os.path.join(
            os.getcwd(), 'artifacts', 'brain_scores',
            f'{date}_layer={layer_list[0]}_finetuned={finetuned}',
            f'{date}_finetuned={finetuned}.csv'), index=False,
            sep=',')

        # Create a text file and save it to path_to_brain_scores with metadata
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

            # load metadata from checkpoint as well
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

                df = pd.DataFrame((coeff_of_det_per_subject[i], ridge_regress_predict_per_subject[i])).to_csv(
                    os.path.join(
                        path_to_brain_scores, f'subject_{i}_brain_scores_per_feature.csv'),
                    index=False, sep=',')
        # print("coeff of det per subject: ", coeff_of_det_per_subject, "\nridge regress per subject: ", ridge_regress_predict_per_subject)


    # instantiating a model here. For now this is a base model. Will want to look at how to load a checkpoint with
    # the new pl_model wrapper we have now.

    # TODO investigate why the following line was necessary in the old before this. Artyom fixed this to make it automatic
    # but still don't fully understand it
    # Warning - training head dims should match difumo resolution
    # train_head_dims = [2, 1024]

    """
    Regarding the heads. Yes, pulling apart the pl_model class has heads set up to go to fmri dataset and the hf dataset.
    fMRI dataset has dimension 1024 and hf dataset has dimensionality 2. For the life of me I cannot remember why we
    need a head with dimendata_module = BrainBiasDataModule(context.get_ds_configs(), tokenizer)sionality of 2. Is it my myopia due to focusing only on brain scores?
    Split into acceptable or unacceptable by Hendrycks.


    Anyways, to get this for each layer use the forward function of the pl_model and pass it the tokens and attention
    mask, as usual. In return, get the results.
    However, he has it set up for a specific token location, i.e. to grab the CLS token or something.
    But I want to have it run over all the tokens.
    I need to step through the different calculations in a single call of forward and see the dimensionality of
    the model output at each step. 

    In order to test that, first need to load some data and get some sample tokens and attention masks.
    Load in data in batches as I specify. Then can get 1/8 or whatever with my batch size for ridge regression.
    """

    # base model
    model = AutoModel.from_pretrained(context.model_path)
    tokenizer = AutoTokenizer.from_pretrained(context.model_path)
    data_module = BrainBiasDataModule(context.get_ds_configs(), tokenizer)
    base_model = PLModel(model, context.plc, data_module)

    from datasets import load_dataset

    print(os.getcwd())

    ds = load_dataset('/home/austin/PycharmProjects/Inducing-human-like-biases-in-moral-reasoning-LLMs/data/ds000212/ds000212_lfb', name='LFB-LAST')

    sub07_test_list = list(ds.filter(lambda e: 'sub-07' in e['file'])['train'])

    dataloader = DataLoader(ds['train'], batch_size=8)
    for batch in dataloader:
        activations = batch['label']
        input = batch['input']
        break


    # finetuned model
    # model = AutoModel.from_pretrained(context.model_path)
    # tokenizer = AutoTokenizer.from_pretrained(context.model_path)
    #
    # finetuned_model = load_from_checkpoint(model, context)
    # data_module = BrainBiasDataModule(context.get_ds_configs(), tokenizer)
    # finetuned_model = PLModel(finetuned_model, context.plc, data_module)

    # now, just pass n different models to the wrapper function to finetune or not.
    # no need to load them each time within the wrapper.

    """
    As per above, now testing dataloaders to get sample token and attention mask.
    """
    #
    # data_module = BrainBiasDataModule(context.get_ds_configs(), tokenizer)
    # combined_dataloader = data_module.train_dataloader()
    #
    # # fmri_dataset = combin
    # for batch, samples in enumerate(combined_dataloader):
    #     ethics_scores, fmri = samples
    #     break

    print()

    # commented out for now to look at model's internals in debugger without wrecking my old laptop
    # #base BERT has 12 encoder layers
    # layer_list = ['2','3','4','5','6','7','8','9','10','11']#,'12'] #including layer 1 and 12 breaks it for some reason
    # path_to_checkpoint = return_path_to_latest_checkpoint()
    # date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") #for naming
    # #wrapper(path_to_model, layer_list, finetuned=False)
    #
    # #instead of using layer list, just pass a single layer to the wrapper function and loop using that.
    # #will need to incorporate the layer into the directory name then.
    # for i in layer_list:
    #     wrapper(path_to_checkpoint, [i], date, finetuned=False)
    #     wrapper(path_to_checkpoint, [i], date, finetuned=True)
    #     #break
    #
    # #wrapper(path_to_model, ['2'], date, finetuned=False)
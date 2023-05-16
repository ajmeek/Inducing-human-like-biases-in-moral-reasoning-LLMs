from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.model import BERT
from src.utils.loading_data import load_ds000212_dataset

from sklearn.linear_model import RidgeCV


def calculate_brain_scores(model: nn.Module,
                           model_input: torch.tensor,
                           test_data: torch.tensor,
                           epochs_to_train_head: int):

    # print(model)
    activations = {}
    layers = ['3', '4', '5']
    modules = ['intermediate.dense']
    all_layer_names = []
    for layer in layers:
        for module in modules:
            all_layer_names.append(f'{layer}.{module}')

    def get_activation(name):
        def hook(intermediate_model, input, output):
            if any(x in name for x in all_layer_names):
                activations[name] = output.detach()

            # The below code might be needed depending on the layers we want to inspect.
            # if isinstance(output, tuple) and len(output) == 1:
            #     activations[name] = output[0].detach()
            # elif isinstance(output, BaseModelOutputWithPastAndCrossAttentions) or isinstance(output, BaseModelOutputWithPoolingAndCrossAttentions):
            #     pass  # TODO: not sure what happens here, have to check.
            # elif isinstance(intermediate_model, model.__class__):  # Ths is the whole model with as output the last layer. Which is not interesting for us.
            #     pass
            # else:

        return hook

    # Attach the hook to every layer
    for name, layer in model.named_modules():
        layer.register_forward_hook(get_activation(name))

    # Run the model
    model.eval()
    with torch.no_grad():
        tokens = torch.tensor(model_input['input_ids'])
        attention_mask = torch.tensor(model_input['attention_mask'])
        output = model(tokens, attention_mask)

    # Calculate the brain scores
    brain_scores = {}
    for layer_name, activation in activations.items():
        print('Calculating brain score for layer:', layer_name,
              'and activation dims: ', activation.shape, '...')
        activations_flattened = activation.flatten().unsqueeze(0)

        clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(activations_flattened, test_data[:, 1000])
        print(clf.score(activations_flattened, test_data[:, 1000]))

        # class LinearClass(nn.Module):
        #     def __init__(self, in_features: int, out_features: int):
        #         super().__init__()
        #         self.linear_1 = nn.Linear(in_features, 50)
        #         self.linear_2 = nn.Linear(50, out_features)
        #
        #     def forward(self, x):
        #         x = self.linear_1(x)
        #         x = self.linear_2(x)
        #         return x
        #
        # linear_model = LinearClass(activations_flattened.shape[1], test_data.shape[1])
        # loss_function = nn.MSELoss()
        # optimizer = optim.SGD(linear_model.parameters(), lr=1e-6)
        #
        # Calculate the brain score by training the linear layer on the test data.
        # for i in tqdm(range(epochs_to_train_head)):
        #     # output = linear_model(activations_flattened)
        #     # loss = loss_function(output, test_data)
        #
        #     # optimizer.zero_grad()
        #     # loss.backward()
        #     # optimizer.step()
        #     print(loss)
        #
        # brain_scores[layer_name] = loss.item()

    return brain_scores


if __name__ == '__main__':
    # Specify parameters
    checkpoint_name = 'bert-base-cased'
    path_to_model = r'..\artifacts\230515-101547\model.pt'  # Specify the path to the model.
    train_head_dims = [2, 39127]  # Need to fill this in to not get an error when loading the model. This is not used in the brain score calculation.

    epochs_to_train_head = 50

    # Load the model
    base_model = AutoModel.from_pretrained(checkpoint_name)
    model = BERT(base_model, head_dims=train_head_dims)
    model.load_state_dict(torch.load(path_to_model))

    # Load the data
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
    # path_to_data = r'C:..\data\ethics\train.csv'
    model_input = 'I told my baby I loved her when she cried.'  # Example of the commonsense test csv.
    tokenized = tokenizer([model_input], padding='max_length', truncation=True)

    test_data_path = Path(r'..\data')
    test_data = load_ds000212_dataset(test_data_path, tokenizer, num_samples=1, normalize=True)[2]

    # Calculate the brain scores
    brain_scores = calculate_brain_scores(model, tokenized, test_data, epochs_to_train_head)
    print(brain_scores)

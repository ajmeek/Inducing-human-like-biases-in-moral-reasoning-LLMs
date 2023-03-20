# This is from the colab notebook of Bogdan
import torch
import torch.nn as nn
import torch.nn.functional as F

import re


# see https://github.com/r-three/t-few/blob/master/src/models/lora.py#L60
def modify_with_ia3(transformer: nn.Module, layers_to_replace: str) -> nn.Module:
    """
    Replace the linear layers in the transformer with IA3Linear layers.
    """
    for m_name, module in transformer.named_modules():
        if any([ia3_layer in m_name for ia3_layer in layers_to_replace.split('|')]):
            if isinstance(module, nn.Linear):  # only replace linear layers

                # Get parent module
                current_module_tree = m_name.split('.')
                all_parent_modules = current_module_tree[:-1]
                parent_module = transformer
                for i in all_parent_modules:
                    parent_module = getattr(parent_module, i)

                # Replace linear layer with IA3Linear in the parent module of the linear layer.
                current_module_name = current_module_tree[-1]
                setattr(parent_module, current_module_name, IA3Linear(module))

    return transformer


# see https://github.com/r-three/t-few/blob/master/src/models/lora.py#L35
class IA3Linear(nn.Module):
    """
    The IA3Linear layer is a wrapper around a linear layer that is used to element-wise multiply (i.e. rescale) the model's activations against a learned vector.
    https://arxiv.org/pdf/2205.05638.pdf section 3.3 Parameter-efficient fine-tuning with IA3
    """
    def __init__(self, linear_layer: nn.Linear):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        # see https://github.com/r-three/t-few/blob/master/configs/ia3.json for default config values
        self.rank = 0
        self.scaling_rank = 1
        self.init_scale = 0.

        self.weight = linear_layer.weight
        self.bias = linear_layer.bias

        # parsimonious implementation of what would be executed in the ifs from https://github.com/r-three/t-few/blob/master/src/models/lora.py#L16
        # doesn't seem used anyway for IA3
        '''
        self.multi_lora_a = nn.Parameter(
                torch.ones(self.scaling_rank, linear_layer.in_features)
                + torch.randn(self.scaling_rank, linear_layer.in_features) * self.init_scale
        )
        '''

        self.multi_lora_b = nn.Parameter(torch.ones(linear_layer.out_features, self.scaling_rank))

    # see https://github.com/r-three/t-few/blob/master/src/models/lora.py#L36 and
    # https://github.com/r-three/t-few/blob/master/configs/ia3.json (only *lora_b.* trainable/requires_grad and multi_lora)
    def forward(self, input):
        # parsimonious implementation for ia3 (notice that multi_lora_a is not trainable - https://github.com/r-three/t-few/blob/master/configs/ia3.json)
        hidden = F.linear(input, self.weight, self.bias)
        if self.multi_lora_b.requires_grad:
            hidden = hidden * self.multi_lora_b.flatten()
        return hidden

    # see https://github.com/r-three/t-few/blob/master/src/models/lora.py#L54
    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}, rank={}, scaling_rank={}".format(
            self.in_features, self.out_features, self.bias is not None, self.rank, self.scaling_rank
        )


if __name__ == '__main__':
    from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, RobertaConfig, RobertaTokenizer, RobertaModel

    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')  # ('bert-base-cased')
    config = AutoConfig.from_pretrained('roberta-large', num_labels=1)
    old_model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased')         #('roberta-large')
    # old_model = RobertaModel.from_pretrained('models/cm_roberta-large.pt', local_files_only=True, config=config)

    text = "Replace me by any text you'd like."

    with torch.no_grad():
        encoded_input = tokenizer(text, return_tensors='pt')
        old_output = old_model(**encoded_input)

    layers_to_replace = "key|value|intermediate.dense"

    print('old model')
    print(old_model)

    model = modify_with_ia3(old_model, layers_to_replace)

    print('new model')
    print(model)

    with torch.no_grad():
        new_output = model(**encoded_input)

    # These outputs should be the same
    print(old_output)
    print(new_output)

    print("Trainable parameters")
    trainable_param_names = ".*lora_b.*"
    print(
        [
            p_name
            for p_name in dict(model.named_parameters()).keys()
            # if re.fullmatch(trainable_param_names, p_name)
        ]
    )
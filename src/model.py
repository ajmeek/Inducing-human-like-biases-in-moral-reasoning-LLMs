import math
from typing import Union
import torch.nn as nn

# model including the base and multiple heads
# the heads are specified by the head_dims argument - the dimensionality of
# each head can be an int or a tuple of ints
class BERT(nn.Module):
    def __init__(self, base_model, head_dims: list[Union[int, tuple[int]]] = (2,)):
        super().__init__()
        self.base = base_model

        # initialize all the heads
        # if the desired output has multiple axes, we want to output it flattened
        # and then reshape it at the end
        self.head_dims = head_dims
        head_in_dim = base_model.config.hidden_size

        heads = []
        for head_d in head_dims:
            # For now, we make everything a flattened 1D output
            head_d_flat = math.prod(head_d) if type(head_d) is tuple else head_d
            heads.append(nn.Linear(head_in_dim, head_d_flat))
        self.heads = nn.ModuleList(heads)

    def forward(self, tokens, mask):
        # tokens: [batch seq_len]
        # mask: [batch seq_len]
        base_out = self.base(tokens, mask)  # [batch seq_len d_model]
        base_out = base_out.last_hidden_state  # use last layer activations
        base_out = base_out[:, 0, :]  # only take the encoding of [CLS] -> [batch, d_model]

        outs = []
        for head, head_d in zip(self.heads, self.head_dims):
            head_out = head(base_out)  # [batch d_out_flat]

            # if out_dim is multidimensional, reshape the output
            if type(head_d) is tuple:
                d_batch = head_out.shape[0]
                # Unflatten the output again, note that targets are also not flat.
                head_out = head_out.reshape((d_batch, *head_d))  # [batch *head_d]

            outs.append(head_out)

        return outs

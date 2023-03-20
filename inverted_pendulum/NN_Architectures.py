import torch.nn as nn
import torch
import numpy as np


class SimpleLinearNN(nn.Module):
    def __init__(self, input_dim, output_dim, hdim=20, activation_fn=nn.ReLU(), num_layers=3):
        super(SimpleLinearNN, self).__init__()

        # Create the actual architecture
        model = []
        curr_dim = input_dim
        for _ in range(num_layers-1):
            model.append(nn.Sequential(
                                nn.Linear(curr_dim, hdim),
                                activation_fn))
            curr_dim = hdim
        model.append(nn.Sequential(nn.Linear(hdim, output_dim)))
        self.model = nn.Sequential(*model)
        self.float()

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model = SimpleLinearNN(3, 2)
    print(model)

    rand_tensor = torch.rand((512, 3))
    print(rand_tensor)
    print(model(rand_tensor).shape)
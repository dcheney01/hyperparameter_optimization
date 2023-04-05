import torch.nn as nn
import torch

def SimpleLinearNN(input_dim, output_dim, num_hidden_layers=3, hdim=20, activation_fn=nn.ReLU()):
    """
    Function that returns a Linear Model with 2 + num_hidden_layers total layers
        - Each layer besides the first and last have hdim hidden nodes
        - Each layer besides the last will use the given activation function 
    """
    model = []
    model.append(nn.Sequential(nn.Linear(input_dim, hdim), activation_fn))
    for _ in range(num_hidden_layers):
        model.append(nn.Sequential(
                            nn.Linear(hdim, hdim),
                            activation_fn)
                    )
    model.append(nn.Sequential(nn.Linear(hdim, output_dim)))
    model = nn.Sequential(*model)

    return model


if __name__ == '__main__':
    model = SimpleLinearNN(3, 2, num_hidden_layers=0)
    print(model)

    rand_tensor = torch.rand((512, 3))
    print(model(rand_tensor).shape)
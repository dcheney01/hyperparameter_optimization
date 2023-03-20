import torch.nn as nn
import torch

class SimpleLinearNN(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers=3, hdim=20, activation_fn=nn.ReLU(), ):
        super(SimpleLinearNN, self).__init__()

        # Create the actual architecture
        model = []
        model.append(nn.Sequential(nn.Linear(input_dim, hdim), activation_fn))
        for _ in range(num_hidden_layers):
            model.append(nn.Sequential(
                                nn.Linear(hdim, hdim),
                                activation_fn))
        model.append(nn.Sequential(nn.Linear(hdim, output_dim)))

        self.model = nn.Sequential(*model)
        self.float()

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model = SimpleLinearNN(3, 2, num_hidden_layers=0)
    print(model)

    rand_tensor = torch.rand((512, 3))
    print(rand_tensor)
    print(model(rand_tensor).shape)
import torch.nn as nn
import torch
from torch.autograd import Variable 
import sys
import math
sys.path.append('/home/daniel/research/catkin_ws/src/')


def SimpleLinearNN(input_dim, output_dim, num_hidden_layers=3, hdim=20, activation_fn=nn.ReLU()):
    """
    Function that returns a Linear Model with 2 (input/output) + num_hidden_layers total layers
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

    model.apply(init_weights)
    return model

class LSTM_CUSTOM(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hdim, activation_fn=nn.ReLU()):
        super(LSTM_CUSTOM, self).__init__()
        self.num_layers = num_layers #number of layers
        self.input_dim = input_dim #input size
        self.hidden_size = hdim #hidden state

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, batch_first=True) #lstm
        
        self.activation_fn = activation_fn
        self.fc_final = nn.Linear(self.hidden_size, output_dim) #fully connected last layer

        self.lstm.apply(init_weights)
        self.fc_final.apply(init_weights)
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.activation_fn(hn)
        out = self.fc_final(out) #Final Output
        return out

class Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    # Constructor
    def __init__(
        self,
        num_tokens, # number of outputs
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, num_tokens)
        
    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        
        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)
        
        return out
      
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token) 


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


def init_weights(model):
    if isinstance(model, nn.Linear):
        nn.init.uniform_(model.weight)
        nn.init.constant_(model.bias, 0)
    elif isinstance(model, nn.LSTM):
        for child, param in model.named_parameters():
            if 'weight' in child:
                nn.init.uniform_(param)
            elif 'bias' in child:
                nn.init.constant_(param, 0)
            else:
                print(f"Found layer weights that was not bias or weights, {child}")


if __name__ == '__main__':
    # model = SimpleLinearNN(3, 2, num_hidden_layers=0)
    # print(model)
    # rand_tensor = torch.rand((1, 5, 3))
    # print(model(rand_tensor).shape)

    

    # Test the LSTM Class
    rand_tensor = torch.rand((1, 5, 3))
    model = LSTM_CUSTOM(3, 2, 1, 10)
    # for child, param in model.named_parameters():
    #     print(child, param.shape)
    print(model(rand_tensor).shape)


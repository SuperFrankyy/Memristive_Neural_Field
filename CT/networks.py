import os
import numpy as np

import torch
import torch.nn as nn

from scipy.stats import skewnorm

############ Input Positional Encoding ############
class Positional_Encoder():
    def __init__(self, params):
        self.params = params
        if params['embedding'] == 'gauss':
            # Try loading real RRAM device data, fall back to random Gaussian
            data_path = '../../NeRP/lizhi_data_new/processed_data_for_encoding.npy'
            if os.path.exists(data_path):
                self.data = np.load(data_path)
                self.B = torch.Tensor(self.data[0, :self.params['embedding_size']*self.params['coordinates_size']]).reshape(self.params['embedding_size'], self.params['coordinates_size'])
                self.B = (self.B/28 - 1) * 28
                print(f'Loaded RRAM encoding data from {data_path}')
            else:
                print(f'WARNING: {data_path} not found, using random Gaussian encoding')
                self.data = None
                self.B = torch.randn((params['embedding_size'], params['coordinates_size'])) * params['scale']

            print(self.B)
            self.B = self.B.cuda()
        elif params['embedding'] == 'none':
            self.B = None
        elif params['embedding'] == 'basic':
            self.B = None
        elif params['embedding'] == 'positional':
            m = params['embedding_size']
            # B: each row is powers of 2 from 2^0 to 2^(m-1)
            self.B = torch.Tensor([2**i for i in range(m)]).unsqueeze(1).repeat(1, params['coordinates_size'])
            # print(self.B)
            self.B = self.B.cuda()

        # simulated noise distributions for analysis
        elif params['embedding'] == 'RC-Gaussian':
            # Right-Censored Gaussian
            self.B = torch.randn((params['embedding_size'], params['coordinates_size'])) * params['scale']
            threshold = params.get('threshold', 8.0)
            self.B = torch.minimum(self.B, torch.tensor(threshold))
            print(self.B)
            self.B = self.B.cuda()

        elif params['embedding'] == 'LC-Gaussian':
            # Left-Censored Gaussian
            self.B = torch.randn((params['embedding_size'], params['coordinates_size'])) * params['scale']
            threshold = params.get('threshold', -8.0)
            self.B = torch.maximum(self.B, torch.tensor(threshold))
            print(self.B)
            self.B = self.B.cuda()

        elif params['embedding'] == 'RT-Gaussian':
            B_large = torch.randn(params['embedding_size'] * 2 * params['coordinates_size']) * params['scale']
            threshold = params.get('threshold', 8.0)
            B_large = B_large[B_large <= threshold]
            while B_large.shape[0] < params['embedding_size'] * params['coordinates_size']:
                extra_B = torch.randn(params['embedding_size'] * 2 * params['coordinates_size']) * params['scale']
                B_large = torch.cat((B_large, extra_B[extra_B <= threshold]), dim=0)
            self.B = B_large[:params['embedding_size'] * params['coordinates_size']].reshape(params['embedding_size'], params['coordinates_size'])
            print(self.B)
            self.B = self.B.cuda()

        elif params['embedding'] == 'LT-Gaussian':
            B_large = torch.randn(params['embedding_size'] * 2 * params['coordinates_size']) * params['scale']
            threshold = params.get('threshold', -8.0)
            B_large = B_large[B_large >= threshold]
            while B_large.shape[0] < params['embedding_size'] * params['coordinates_size']:
                extra_B = torch.randn(params['embedding_size'] * 2 * params['coordinates_size']) * params['scale']
                B_large = torch.cat((B_large, extra_B[extra_B >= threshold]), dim=0)
            self.B = B_large[:params['embedding_size'] * params['coordinates_size']].reshape(params['embedding_size'], params['coordinates_size'])
            print(self.B)
            self.B = self.B.cuda()

        elif params['embedding'] == 'ideal-Gaussian':
            # Right-Censored Gaussian
            self.B = torch.randn((params['embedding_size'], params['coordinates_size'])) * params['scale']
            # threshold = params.get('threshold', 8.0)
            # self.B = torch.minimum(self.B, torch.tensor(threshold))
            print(self.B)
            self.B = self.B.cuda()

        elif params['embedding'] == 'Right-Skewnorm':
            # Right-Censored Skewnorm
            self.B = skewnorm.rvs(2, size=(params['embedding_size'], params['coordinates_size'])) * params['scale']
            print(self.B)
            self.B = torch.Tensor(self.B).cuda()
        
        elif params['embedding'] == 'Left-Skewnorm':
            # Left-Censored Skewnorm
            self.B = skewnorm.rvs(-2, size=(params['embedding_size'], params['coordinates_size'])) * params['scale']
            print(self.B)
            self.B = torch.Tensor(self.B).cuda()

        else:
            raise NotImplementedError

    def embedding(self, x, iter_num=0):
        if self.params['embedding'] == 'gauss':
            if self.data is not None:
                self.B = torch.Tensor(self.data[0, :self.params['embedding_size']*self.params['coordinates_size']]).reshape(self.params['embedding_size'], self.params['coordinates_size'])
                self.B = (self.B/28 - 1) * 28
                self.B = self.B.cuda()
            x_embedding = (2. * np.pi * x) @ self.B.t()
            # x_embedding + original input
            x_embedding = torch.cat([x, torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)
            # x_embedding = torch.cat([torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)
        elif self.params['embedding'] == 'none':
            x_embedding = x
        elif self.params['embedding'] == 'basic':
            x_embedding = torch.cat([x, torch.sin(x), torch.cos(x)], dim=-1)
        elif self.params['embedding'] == 'positional':
            x_embedding = (2. * np.pi * x) @ self.B.t()
            x_embedding = torch.cat([x, torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)
        elif self.params['embedding'] == 'RC-Gaussian':
            x_embedding = (2. * np.pi * x) @ self.B.t()
            x_embedding = torch.cat([x, torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)
        elif self.params['embedding'] == 'LC-Gaussian':
            x_embedding = (2. * np.pi * x) @ self.B.t()
            x_embedding = torch.cat([x, torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)
        elif self.params['embedding'] == 'RT-Gaussian':
            x_embedding = (2. * np.pi * x) @ self.B.t()
            x_embedding = torch.cat([x, torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)
        elif self.params['embedding'] == 'LT-Gaussian':
            x_embedding = (2. * np.pi * x) @ self.B.t()
            x_embedding = torch.cat([x, torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)
        elif self.params['embedding'] == 'ideal-Gaussian':
            x_embedding = (2. * np.pi * x) @ self.B.t()
            x_embedding = torch.cat([x, torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)
        elif self.params['embedding'] == 'Right-Skewnorm':
            x_embedding = (2. * np.pi * x) @ self.B.t()
            x_embedding = torch.cat([x, torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)
        elif self.params['embedding'] == 'Left-Skewnorm':
            x_embedding = (2. * np.pi * x) @ self.B.t()
            x_embedding = torch.cat([x, torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)
        return x_embedding
 



############ Fourier Feature Network ############
class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class FFN(nn.Module):
    def __init__(self, params):
        super(FFN, self).__init__()

        num_layers = params['network_depth']
        hidden_dim = params['network_width']
        input_dim = params['network_input_size']
        output_dim = params['network_output_size']

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for i in range(1, num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out



############ SIREN Network ############
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / \
            self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)


class SIREN(nn.Module):
    def __init__(self, params):
        super(SIREN, self).__init__()

        num_layers = params['network_depth']
        hidden_dim = params['network_width']
        input_dim = params['network_input_size']
        output_dim = params['network_output_size']
        rank = int(params['rank'])

        layers = [SirenLayer(input_dim, hidden_dim, is_first=True)]
        # for i in range(1, num_layers - 1):
        #     layers.append(SirenLayer(hidden_dim, hidden_dim))
        # layers.append(SirenLayer(hidden_dim, output_dim, is_last=True))

        # LoRA
        if params['rank'] == 0:
            for i in range(1, num_layers - 1):
                layers.append(SirenLayer(hidden_dim, hidden_dim))
        else:
            for i in range(1, num_layers - 1):
                layers.append(SirenLayer(hidden_dim, rank))  # low-rank decomposition
                layers.append(SirenLayer(rank, hidden_dim))
                # layers.append(nn.Dropout(0.1)) # add a dropout layer to prevent overfitting
        layers.append(SirenLayer(hidden_dim, output_dim, is_last=True))
        # layers.append(SirenLayer(hidden_dim + input_dim, output_dim, is_last=True))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
                    
        return out


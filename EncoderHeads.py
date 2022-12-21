import numpy as np
import torch


class FullyConnectedEncoderHead(torch.nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super(FullyConnectedEncoderHead, self).__init__()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, embed_dim),
            torch.nn.Tanh()
        )
        self.num_parameters = sum([np.prod(params.size()) for params in self.fc.parameters()])

    def forward(self, x):
        return self.fc(x)

class FullyConnectedBatchNormEncoderHead(torch.nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super(FullyConnectedBatchNormEncoderHead, self).__init__()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, embed_dim),
            torch.nn.BatchNorm1d(embed_dim)
        )
        self.num_parameters = sum([np.prod(params.size()) for params in self.fc.parameters()])

    def forward(self, x):
        return self.fc(x)

class FullyConnectedDropoutEncoderHead(torch.nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super(FullyConnectedDropoutEncoderHead, self).__init__()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.35),
            torch.nn.Linear(512, embed_dim),
            torch.nn.Tanh()
        )
        self.num_parameters = sum([np.prod(params.size()) for params in self.fc.parameters()])

    def forward(self, x):
        return self.fc(x)

class BiLSTMEncoderHead(torch.nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super(BiLSTMEncoderHead, self).__init__()

        self.lstm_hidden_size = 150
        self.lstm_num_layers = 1
        self.lstm = torch.nn.LSTM(input_size = input_dim[1],
                                  hidden_size=self.lstm_hidden_size,
                                  num_layers=self.lstm_num_layers,
                                  bidirectional=True)
        self.fc = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(2 * self.lstm_hidden_size, embed_dim),
            torch.nn.Tanh()
        )
        self.num_parameters = sum([np.prod(params.size()) for params in self.lstm.parameters()])
        self.num_parameters += sum([np.prod(params.size()) for params in self.fc.parameters()])

    def forward(self, x):
        batch_size = x.shape[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h_0 = torch.zeros(2 * self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        c_0 = torch.zeros(2 * self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        lstm_out, (_, _) = self.lstm(x, (h_0, c_0))
        fc_input = torch.cat((lstm_out[:, -1, :self.lstm_hidden_size], lstm_out[:, 0, self.lstm_hidden_size:]), dim=1)
        # The first tensor corresponds to normal LSTM, the second corresponds to reverse LSTM.
        return self.fc(fc_input)

class IdentityEncoderHead(torch.nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super(IdentityEncoderHead, self).__init__()

        self.fc = torch.nn.Identity()
        self.num_parameters = 0

    def forward(self, x):
        return self.fc(x)

def EncoderHead(name, input_dim, embed_dim=128):
    if name == "FC":
        if isinstance(input_dim, list):
            raise TypeError("Pretrained encoder and encoder head are not compatible.")
        return FullyConnectedEncoderHead(input_dim, embed_dim)
    elif name == "FC_BatchNorm":
        if isinstance(input_dim, list):
            raise TypeError("Pretrained encoder and encoder head are not compatible.")
        return FullyConnectedBatchNormEncoderHead(input_dim, embed_dim)
    elif name == "FC_Dropout":
        if isinstance(input_dim, list):
            raise TypeError("Pretrained encoder and encoder head are not compatible.")
        return FullyConnectedDropoutEncoderHead(input_dim, embed_dim)
    elif name == "BiLSTM":
        if not isinstance(input_dim, list):
            raise TypeError("Pretrained encoder and encoder head are not compatible.")
        return BiLSTMEncoderHead(input_dim, embed_dim)
    elif name == "MeanPooling":
        if not isinstance(input_dim, list):
            raise TypeError("Pretrained encoder and encoder head are not compatible.")
        return MeanPoolingEncoderHead(input_dim, embed_dim)
    elif name == "Identity":
        if isinstance(input_dim, list):
            raise TypeError("Pretrained encoder and encoder head are not compatible.")
        return IdentityEncoderHead(input_dim, embed_dim)
    else:
        raise ValueError(name + " has not been implemented!")

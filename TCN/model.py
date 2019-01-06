import torch
from torch import nn
from TCN.tcn import TemporalConvNet
from tensorboardX import SummaryWriter


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.writer = SummaryWriter('logs')
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1]*10, num_channels[-1]*10)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(num_channels[-1]*10, 512)
        self.activate = nn.Tanh()
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        x, i = x
        output = []
        for xx in x:
            output.append(torch.mean(self.tcn(xx), dim=2))
        output = torch.cat(tuple(output), dim=1)
        att = self.sig(self.linear(output))
        output = att*output
        output0 = self.fc1(output)
        output = self.activate(-output0)

        if i % 1000 == 0:
            self.writer.add_histogram("att", att.cpu().data.numpy(), i)
            self.writer.add_histogram("feat", output0.cpu().data.numpy(), i)
            self.writer.add_histogram("feat_active", output.cpu().data.numpy(), i)

        output = self.fc2(output)

        return output

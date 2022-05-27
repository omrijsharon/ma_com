import torch
from torch import nn
from torch.nn import functional as F
from functools import partial


def max(x, dim=1):
    return x.max(dim=dim)[0]


class Macom(nn.Module):
    def __init__(self, encoding_net, decoding_net, reduce="max"):
        super(Macom, self).__init__()
        self.enc = encoding_net
        self.dec = decoding_net
        if reduce == "mean":
            self.reduce_fn = partial(torch.mean, dim=1)
        elif reduce == "max":
            self.reduce_fn = partial(max, dim=1)
        else:
            raise ValueError("reduce must be either mean or max")

    def forward(self, x):
        """
        :param x: communication messages from n_agents in the form [batch, n_agents, latent_msg_dim]
        :return: messages concept in a latent space in the form [batch, msg_dim]
        """
        enc_x = self.enc(x)
        w = torch.bmm(enc_x, enc_x.transpose(1, 2))
        w = F.softmax(w, dim=1)
        pre_output = torch.bmm(w, enc_x)
        output = self.dec(pre_output)
        return self.reduce_fn(output)

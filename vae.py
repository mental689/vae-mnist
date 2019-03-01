import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(28*28, 20*20)
        self.fc_mu = nn.Linear(20*20, 20)
        self.fc_std = nn.Linear(20*20, 20)
        self.fc2 = nn.Linear(20, 20*20)
        self.fc3 = nn.Linear(20*20, 28*28)

    def encode(self, x):
        x = F.relu(self.fc1(x))
        return self.fc_mu(x), self.fc_std(x)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(z))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar



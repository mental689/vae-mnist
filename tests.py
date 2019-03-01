import unittest, os, sys
sys.path.insert(0, os.path.abspath('.'))
import torch
torch.manual_seed(12345)
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
from tensorboardX import SummaryWriter

from vae import VAE

import logging
logger = logging.getLogger(__name__)


def vae_loss(y, x, mu, logvar):
    return ((y-x.view(-1,784))**2).sum() - 0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())

def train(epoch, model, optimizer, dataloader, writer):
    train_loss = 0.
    model.train(True)
    pbar = tqdm(dataloader)
    num_iter = 0
    # max_iter = len(dataloader)
    for (data, _) in pbar:
        num_iter += 1
        optimizer.zero_grad()
        y, mu, logvar = model(data)
        loss = vae_loss(y, data, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pbar.set_description('Epoch {}, iter {}, loss={:.2f}'.format(epoch,
            num_iter, train_loss / num_iter))
    writer.add_scalar('Train loss', train_loss / num_iter, epoch)


def test(epoch, model, dataloader, writer):
    model.train(False)
    model.eval()
    test_loss = 0.
    with torch.no_grad():
        pbar = tqdm(dataloader)
        num_iter = 0
        for (data, _) in pbar:
            num_iter += 1
            y, mu, logvar = model(data)
            test_loss += vae_loss(y, data, mu, logvar)
            pbar.set_description('Epoch {}, test iter {}, loss {:.2f}'.format(epoch, num_iter, test_loss /
                        len(dataloader)))
            if num_iter == 1:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      y.view(128, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    writer.add_scalar('Test loss', test_loss / len(dataloader), epoch)


class TestVAE(unittest.TestCase):
    def setUp(self):
        self.model = VAE()

    def testDummy(self):
        logger.info(self.model)

    def testTrainMNIST(self):
        # Setup dataset
        train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()), batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data',
            train=False, transform=transforms.ToTensor()),
            batch_size=128, shuffle=True)
        # Optimizers
        optimizer = optim.Adam(self.model.parameters(), lr=1e-03)
        if not os.path.exists('./log'):
            os.makedirs('./log')
        writer = SummaryWriter(log_dir='./log')

        # Train and test
        for epoch in range(1000):
            train(epoch, self.model, optimizer, train_loader, writer)
            test(epoch, self.model, test_loader, writer)
            with torch.no_grad():
                sample = torch.randn(64, 20)
                sample = self.model.decode(sample).cpu()
                save_image(sample.view(64, 1, 28, 28),
                        'results/sample_' + str(epoch) + '.png')


if __name__ == '__main__':
    unittest.main()

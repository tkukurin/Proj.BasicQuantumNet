
import data
import util
from modules import Model

import sys
import click
import gin
import logging
import typing as T
import itertools as it

import numpy as np
import torch
import tqdm

from torch import nn
from torch.nn import functional as F
from torch.utils import data as torchdata
from torch import optim

from pathlib import Path


logging.basicConfig(
  format='[%(levelname)s:%(asctime)s] %(message)s',
  datefmt='%m%d@%H%M',
  handlers=[logging.FileHandler('log.txt','a'), logging.StreamHandler(sys.stdout)],
  level=logging.INFO)
L = logging.getLogger(__name__)


class Trainer:
  def __init__(self, model, loss_fn, optimizer, vocab, device):
    self.vocab = vocab
    self.model = model
    self.loss_fn = loss_fn
    self.optimizer = optimizer
    self.device = device

  def _preprocess(self, s1, s2, labels):
    s1 = self.vocab.encode(s1).to(self.device)
    s2 = self.vocab.encode(s2).to(self.device)
    labels = labels.to(self.device)
    return s1, s2, labels

  def train(self, train_loader):
    self.model.train()
    self.model.to(self.device)
    loss_ = 0
    for idx, batch in enumerate(tqdm.tqdm(train_loader)):
      s1, s2, labels = self._preprocess(*batch)
      self.optimizer.zero_grad()
      confs = self.model(s1, s2).to(self.device)
      loss = self.loss_fn(confs, labels).to(self.device)
      loss.backward()

      loss_ += loss.item()
      self.optimizer.step()
    return {'loss': loss_ / len(train_loader)}

  def eval_(self, dev_loader):
    self.model.eval()
    self.model.to(self.device)
    dev_acc = 0
    loss_ = 0
    with torch.no_grad():
      for idx, batch in enumerate(tqdm.tqdm(dev_loader)):
        s1, s2, labels = self._preprocess(*batch)
        confs = self.model(s1, s2)
        preds = F.softmax(confs, dim=-1).argmax(-1)
        loss = self.loss_fn(confs, labels).to(self.device)
        loss_ += loss.item()
        dev_acc += (preds == labels).int().sum().item()

      n_items = (idx * dev_loader.batch_size)
      dev_acc = dev_acc / n_items
      loss_ = loss_ / n_items
    return {'accuracy': dev_acc, 'loss': loss_}

  def train_loop(self, data_dir, n_epochs, model_file, scheduler=None):
    best_loss = 9999
    train_loader = torchdata.DataLoader(
      data.SentPairData(data_dir/'train.tsv'),
      batch_size=64,
      shuffle=True,
      num_workers=2,)
    traces = []
    with tqdm.trange(1, n_epochs + 1) as t:
      for epoch in t:
        train_metrics = self.train(train_loader)
        traces.append({'train': train_metrics})
        # TODO move this?
        with data.SentPairStream(data_dir/'dev.tsv') as dev_data:
          dev_loader = torchdata.DataLoader(dev_data, shuffle=False, batch_size=8)
          dev_metrics = self.eval_(dev_loader)
          traces[-1]['dev'] = dev_metrics

        if scheduler:
          scheduler.step(epoch) #dev_metrics['loss'])

        if epoch % 10 == 0:
          #L.info('Loading best model')
          self.model.load_state_dict(torch.load(model_file))

        avg_loss = train_metrics['loss']
        if avg_loss < best_loss:
          #L.info('Saving best model')
          best_loss = avg_loss
          torch.save(self.model.state_dict(), model_file)

        t.set_postfix(
          best=best_loss,
          avg=avg_loss,
          dev_acc=dev_metrics['accuracy'])
    return traces


@gin.configurable
def main(
    hidden_dim_sweep=(5,10,25),
    n_epochs=100,
    model_file='model.pt',
    data_dir=Path('data'),
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):

  tracess = []
  for hidden_dim in hidden_dim_sweep:
    vocab = util.Vocab.load(data_dir/'vocab.txt', hidden_dim)
    model = Model(
        hidden_dim=hidden_dim,
        out_dim=2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.05,
        verbose=True)
    traces = Trainer(model, loss_fn, optimizer, vocab, device).train_loop(
        data_dir=data_dir,
        n_epochs=n_epochs,
        model_file=model_file,
        scheduler=None)
    L.info('For hidden_dim=%s', hidden_dim)
    L.info('Traces: %s', traces[-2:])
    tracess.append(traces[-2:])
  L.info('Tracess:\n%s', '\n'.join(map(str, tracess)))


if __name__ == '__main__':
  seed = 42
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  util.run_with_gin_config(main, required=False)


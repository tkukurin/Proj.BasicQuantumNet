
import data
import util
import plotting
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


logging.basicConfig(**util.logconf)
L = logging.getLogger(__name__)


@click.group()
@util.click_helper.with_gin_config(required=False)
def main():
  pass


class Trainer:
  def __init__(self, model, loss_fn, vocab, device):
    self.vocab = vocab
    self.model = model
    self.loss_fn = loss_fn
    self.device = device

  def _preprocess(self, s1, s2, labels):
    s1 = self.vocab.encode(s1).to(self.device)
    s2 = self.vocab.encode(s2).to(self.device)
    labels = labels.to(self.device)
    return s1, s2, labels

  def train(self, train_loader, optimizer):
    self.model.train()
    self.model.to(self.device)
    loss_ = 0
    for idx, batch in enumerate(tqdm.tqdm(train_loader)):
      s1, s2, labels = self._preprocess(*batch)
      optimizer.zero_grad()
      confs = self.model(s1, s2).to(self.device)
      loss = self.loss_fn(confs, labels).to(self.device)
      loss.backward()
      loss_ += loss.item()
      optimizer.step()
    return {'loss': loss_ / len(train_loader.dataset)}

  def eval_(self, dev_loader):
    self.model.eval()
    self.model.to(self.device)
    dev_acc = 0
    loss_ = 0
    n_items = 0
    with torch.no_grad():
      for idx, batch in enumerate(tqdm.tqdm(dev_loader)):
        s1, s2, labels = self._preprocess(*batch)
        confs = self.model(s1, s2)
        preds = F.softmax(confs, dim=-1).argmax(-1)
        loss = self.loss_fn(confs, labels).to(self.device)
        loss_ += loss.item()
        dev_acc += (preds == labels).int().sum().item()
        n_items += len(labels)

      dev_acc = dev_acc / n_items
      loss_ = loss_ / n_items
    return {'accuracy': dev_acc, 'loss': loss_}

  def train_loop(self, data_dir, n_epochs, optimizer, scheduler=None):
    '''Run training loop for n_epochs.
    Trainer state will be updated to the best model, as evaluated on the dev set.
    '''
    train_loader = torchdata.DataLoader(
      data.SentPairData(data_dir/'train.tsv'),
      batch_size=64,
      shuffle=True,
      num_workers=2,)
    traces = []
    best_model = None
    best_loss = util.INF
    with tqdm.trange(1, n_epochs + 1) as t:
      for epoch in t:
        train_metrics = self.train(train_loader, optimizer)
        traces.append({'train': train_metrics})
        with data.SentPairStream(data_dir/'dev.tsv') as dev_data:
          dev_loader = torchdata.DataLoader(dev_data, shuffle=False, batch_size=8)
          dev_metrics = self.eval_(dev_loader)
          traces[-1]['dev'] = dev_metrics

        if scheduler:
          scheduler.step(epoch)

        dev_loss = dev_metrics['loss']
        if dev_loss < best_loss:
          best_loss = dev_loss
          best_model = self.model.state_dict()

        t.set_postfix(
          tr_loss=train_metrics['loss'],
          dev_loss=dev_loss,
          dev_acc=dev_metrics['accuracy'])
    self.model.load_state_dict(best_model)
    return traces, best_loss


@main.command()
@gin.configurable
def train(
    hidden_dim_sweep=(5,10,25),
    n_epochs=20,
    out_dir='out',
    data_dir='data',
    device=util.device(),
    Optimizer=optim.Adam,
    seed=42):
  out_dir, data_dir = map(Path, (out_dir, data_dir))
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)

  tracess = []
  best_trainer = None
  best_loss = util.INF
  for hidden_dim in hidden_dim_sweep:
    vocab = util.Vocab.load(data_dir/'vocab.txt')
    model = Model(hidden_dim=hidden_dim, vocab=vocab, out_dim=2)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Optimizer(model.parameters(), lr=1e-4)
    trainer = Trainer(model, loss_fn, vocab, device)
    traces, loss_cur = trainer.train_loop(
      data_dir=data_dir,
      n_epochs=n_epochs,
      optimizer=optimizer,
      scheduler=None)
    if loss_cur < best_loss:
      best_trainer = trainer
      best_loss = loss_cur
    tracess.append((hidden_dim, traces))

  out_dir.mkdir(exist_ok=True)
  for h, traces in tracess:
    plotting.plot_traces(
      traces, out=out_dir/f'traces_{h}.png', title=f'Loss,hidden_dim={h}')
    util.jsondump(traces, out_dir/f'traces.dim_{h}.seed_{seed}.json')

  L.info('Best model loss: %s', best_loss)

  model_file = out_dir/'model.pt'
  L.info('Saving best model to %s', model_file)
  torch.save(best_trainer.model.state_dict(), model_file)


@main.command()
@gin.configurable
def test(
    hidden_dim=25,
    out_dir='out',
    data_dir='data',
    device=util.device()):
  out_dir, data_dir = map(Path, (out_dir, data_dir))
  vocab = util.Vocab.load(data_dir/'vocab.txt')
  model = Model(hidden_dim=hidden_dim, vocab=vocab)
  model.load_state_dict(torch.load(out_dir/'model.pt'))
  loss_fn = nn.CrossEntropyLoss()
  trainer = Trainer(model, loss_fn, vocab, device)

  with data.SentPairStream(data_dir/'dev.tsv') as dev_data:
    dev_loader = torchdata.DataLoader(dev_data, shuffle=False, batch_size=8)
    dev_metrics = trainer.eval_(dev_loader)
    L.info('Dev performance: %s', dev_metrics)

  with data.SentPairStream(data_dir/'test.tsv') as test_data:
    test_loader = torchdata.DataLoader(test_data, shuffle=False, batch_size=8)
    test_metrics = trainer.eval_(test_loader)
    L.info('Test performance: %s', test_metrics)


if __name__ == '__main__':
  main()


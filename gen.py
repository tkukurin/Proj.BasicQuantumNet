
import util

import sys
import click
import gin
import logging
import typing as T
import itertools as it

from nltk.parse.generate import generate
from nltk import CFG
from pathlib import Path

from sklearn.model_selection import train_test_split

from collections import Counter


logging.basicConfig(**util.logconf)
L = logging.getLogger(__name__)

TNameToSents = T.Mapping[str, T.List]


@gin.configurable
def grammar_get(
    base_grammar_fname: util.type.TFile,
    production_fnames: T.List[util.type.TFile],
    sents_per_production=100) -> TNameToSents:
  with open(base_grammar_fname, 'r') as f:
    base = f.read()
  sents = {}
  for production_fname in production_fnames:
    with open(production_fname, 'r') as f:
      productions = f.read()
    grammar = CFG.fromstring(base + productions)
    sents[production_fname] = list(
        generate(grammar, n=sents_per_production, depth=5))
  return sents


@gin.configurable
def output(
    output_dir: util.type.TFile,
    class_to_sents: TNameToSents,
    separator='\t',
    train_size=0.8):
  sents = it.chain(*[
    [(' '.join(s), name) for s in sents]
    for name, sents in class_to_sents.items()
  ])
  sents = it.combinations(sents, 2)
  sents_train, sents_test = train_test_split(
    [(s1, s2, int(l1 == l2)) for (s1, l1), (s2, l2) in sents],
    train_size=train_size)
  sents_dev, sents_test = train_test_split(sents_test, train_size=0.5)

  L.info('Outputting (train=%s,dev=%s,test=%s) sentences to %s',
      len(sents_train), len(sents_dev), len(sents_test), output_dir)
  output_dir = Path(output_dir)
  output_dir.mkdir(exist_ok=True)

  for sents_what, where in [
      (sents_train, 'train.tsv'),
      (sents_dev, 'dev.tsv'),
      (sents_test, 'test.tsv')]:
    with open(output_dir / where, 'w') as f:
      for s1, s2, label in sents_what:
        f.write(f'{s1}{separator}{s2}{separator}{label}\n')

  tokenized_sentlist = it.chain(*class_to_sents.values())
  util.Vocab.save(output_dir / 'vocab.txt', tokenized_sentlist, separator=separator)


def main():
  class_to_sents = grammar_get()
  output(class_to_sents=class_to_sents)


if __name__ == '__main__':
  util.run_with_gin_config(main)


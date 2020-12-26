
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
    domain_grammar_fnames: T.List[util.type.TFile],
    sents_per_domain=100) -> TNameToSents:
  with open(base_grammar_fname, 'r') as f:
    base = f.read()
  sents = {}
  for domain_fname in domain_grammar_fnames:
    with open(domain_fname, 'r') as f:
      productions = f.read()
    grammar = CFG.fromstring(base + productions)
    sents[domain_fname] = list(generate(grammar, n=sents_per_domain))
  return sents


@gin.configurable(blacklist=['class_to_sents'])
def output(
    output_dir: util.type.TFile,
    class_to_sents: TNameToSents,
    separator='\t',
    train_size=0.8):
  sents = list(it.chain(*[
    [(' '.join(s), name) for s in sents]
    for name, sents in class_to_sents.items()
  ]))

  sents_train, sents_test = train_test_split(
    sents, train_size=train_size, shuffle=True)
  sents_dev, sents_test = train_test_split(sents_test, train_size=0.5)

  gen_samples = lambda xs: [
    (s1, s2, int(l1 == l2)) for (s1, l1), (s2, l2) in it.combinations(xs, 2)]
  sents_train, sents_dev, sents_test = map(
    gen_samples, (sents_train, sents_dev, sents_test))

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
  util.Vocab.build(tokenized_sentlist).save(output_dir/'vocab.txt')


@click.command()
@util.click_helper.with_gin_config(required=False)
def main():
  class_to_sents = grammar_get()
  output(class_to_sents=class_to_sents)


if __name__ == '__main__':
  main()


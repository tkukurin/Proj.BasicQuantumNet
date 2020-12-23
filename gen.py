
import sys
import click
import gin
import logging
import typing as T
import itertools as it

from nltk.parse.generate import generate
from nltk import CFG
from pathlib import Path


logging.basicConfig(
  format='[%(levelname)s:%(asctime)s] %(message)s',
  datefmt='%m%d@%H%M',
  handlers=[logging.FileHandler('log.txt','a'), logging.StreamHandler(sys.stdout)],
  level=logging.INFO)
L = logging.getLogger(__name__)

TFile = T.Union[str, Path]


@gin.configurable
def grammar_get(
    base: TFile,
    production_fnames: T.List[TFile],
    sents_to_generate=100):
  with open(base, 'r') as f:
    base = f.read()
  sents = {}
  for production_fname in production_fnames:
    with open(production_fname, 'r') as f:
      productions = f.read()
    grammar = CFG.fromstring(base + productions)
    sents[production_fname] = list(
        generate(grammar, n=sents_to_generate, depth=5))
  return sents


@gin.configurable
def output(output_file: TFile, sentences: dict, SEP='\t'):
  sents = it.chain(*[
    [(' '.join(s), name) for s in sents]
    for name, sents in sentences.items()
  ])

  L.info('Storing 2-combinations of %s sentences to %s', len(sents), output_file)
  with open(output_file, 'w') as f:
    for (s1, l1), (s2, l2) in it.combinations(sents, 2):
      f.write(f'{s1}{SEP}{s2}{SEP}{int(l1 == l2)}\n')


@click.command()
@click.argument('config', type=click.Path(exists=True, dir_okay=False))
@click.pass_context
def cli(ctx: click.Context, config):
  gin.parse_config_file(config)
  sentences = grammar_get()
  output(sentences=sentences)


if __name__ == '__main__':
  cli()


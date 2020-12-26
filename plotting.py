
import util
import click

import numpy as np
import typing as T
import matplotlib.pyplot as plt


# traces are a list of epoch->variant->loss_type
TTraces = T.List[T.Mapping[str,T.Mapping[str,T.Any]]]


def plot_traces(
    traces: T.Union[TTraces, T.List[TTraces]],
    metric_key='loss',
    dataset_keys=['train','dev'],
    out=None,
    title=None):
  if isinstance(traces[0], dict):
    traces = [traces]  # convert to List[TTraces]

  def _datas(variant):
    ys = np.stack([
      np.array([d[variant][metric_key] for d in trace])
      for trace in traces
    ])
    ymean = np.mean(ys, axis=0)
    yerr = np.sum((ys - ymean)**2, axis=0)
    return ymean, yerr

  xs = np.arange(len(traces[0]), dtype=int)
  plt.figure(tight_layout=True)
  if title: plt.title(title)
  for key in dataset_keys:
    ym, ye = _datas(key)
    plt.errorbar(xs, ym, yerr=ye)
  plt.legend(dataset_keys)

  return _plotsave(out)


def _plotsave(out=None):
  if out is not None:
    plt.savefig(out)
  else:
    plt.show()


@click.command()
@util.click_helper.with_path('--out_dir')
def make_aggregate_plot_over_all_seeds(out_dir):
  tracess = []
  for traces_fname in out_dir.glob('traces*seed*.json'):
    tracess.append(util.jsonload(traces_fname))
  plot_traces(tracess, out=out_dir/'traces_aggregate.png')


if __name__ == '__main__':
  make_aggregate_plot_over_all_seeds()




# Dependencies

Honestly, I wasn't using an environment for development because all dependencies
were already installed on my system and this was faster.
In a production situation I'd use `pipenv`, but right now let me just list
dependencies by saying I am using Python 3.9 and rely on latest versions of:
* `matplotlib` (plotting)
* `numpy` (plotting)
* `PyTorch` (optimization)
* `nltk` (grammar generation)
* `scikit-learn` (train and test splits)
* `gin` (config management)
* `click` (config management)

Realistically only a few of these libraries are *strictly* necessary.

# Running the code

Code relies on Gin for config management.
See `config/` for example configurations.

## Generating the dataset

Use `generate.py` with a corresponding grammar.
See `grammar/` for examples:
```
python generate.py --config config/gen_config.gin
```

## Running training / testing

Call `run_traintest.py` with the corresponding parameters:
```
python run_traintest.py --config config/traintest_config.gin train
python run_traintest.py --config config/traintest_config.gin test
```

## Plotting

If you run the training loop with multiple seeds, use `plotting.py` to generate
train/dev loss traces on the output folder like so:
```
python plotting.py --out_dir out/
```


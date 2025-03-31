# Conda Environment

The Conda virtual environment was created using:

```zsh
conda create --prefix ./.dgtrader-venv python=3.10
```

Note that Python 3.10 was chosen since `gym` library is not compatible with Python 3.11 or newer.

```zsh
conda activate ./.dgtrader-venv
conda config --add channels conda-forge # May not be needed if channel already added
conda install ipykernel # Needed for Notebook support
conda env export --no-builds > environment.yml
pip install quantconnect-stubs # This is needed to get auto-complete working
pip install quantconnect-lean
```

The following commands must be run from a Host shell versus Python virtual conda environment:

```zsh
lean init
lean create-project alg
lean backtest alg
```

You can remove the environment by

```zsh
conda remove -p ./.dgtrader-venv --all
```

To activate

```zsh
conda activate ./.dgtrader-venv
```

To re-create

```zsh
conda env create -f environment.yml -p ./.dgtrader-venv
```

To update environment file

```zsh
conda env export --no-builds > environment.yml
```

To download Docker container with Lean run command

```zsh
lean backtest alg
```

To run local research server with notebook support served from docker container run:

```zsh
lean research .
```

To update Lean engine:

```zsh
lean engine update
```

After an update, do make sure to remove absolute path that were defined inside of the [environment.yml](environment.yml) by deleting `name` and `prefix` elements. See previous git history.

---

In order to obtain the csv of one of the stock from [www.alphavantage.co](www.alphavantage.co) source the API key into OS env:

```zsh
source .env
```

The notebooks can be run from VSCode by selecting the environment's kernel.

# Development

The VSCode uses Conda package manager to install needed libraries but the development and backtests run against the docker container. The VSCode intellisense support is provided by

```zsh
pip install quantconnect-stubs # run in the venv
```

The [lean/](./lean/) folder contains all the source code for Lean framework and is imported as Git submodule.

# Pre-requisites

Install following software and note that dev environment setup was only tested on Mac OS.

- [VSCode IDE](https://code.visualstudio.com/download)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Anaconda](https://www.anaconda.com/download)

# Development Environment Setup

- After cloning the repo, open in VSCode and run command `./dev init` to setup Lean Engine, write [.env](.env) file, and add settings to Lean Engine config file [~/.lean/config](~/.lean/config).
- Activate Python virtual environment `conda activate ./.dgtrader-venv`
- Now you should be able to issue command `dev backtest` and can run `dev` CLI without the `./`.


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
# This is needed to get auto-complete working and needs to be in sync with the Lean engine Docker image
pip install quantconnect-stubs==17034 
pip install quantconnect-lean==17034
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

# Lean CLI

[Lean CLI](https://github.com/QuantConnect/lean-cli) is installed and maintained separately from [Lean Engine](https://github.com/QuantConnect/Lean). The CLI uses `engine-image` and `research-image` element key to specify which docker image of [Lean Engine](https://github.com/QuantConnect/Lean) is used for running backtest, live trading, or running Juniper Notebook server. The configuration can be displayed by running command:

```zsh
lean config list
```

The configuration is stored in globally in user folder under `~/.lean/config`. It can be initially set by the following:

```zsh
lean config set engine-image quantconnect/lean:17034
lean config set research-image quantconnect/research:17034
```

# Development

The VSCode uses Conda package manager to install needed libraries but the development and backtests run against the docker container. The VSCode intellisense support is provided by

```zsh
pip install quantconnect-stubs # run in the venv
```

The [lean/](./lean/) folder contains all the source code for Lean framework and is imported as Git submodule.

The [dev](dev) utility is meant to automate most development tasks and can be invoked from the workspace root folder via command:

```zsh
dev clean
```

Currently only `clean` option is supported that removes all backtests from [alg/backtests](alg/backtests/) folder.

The `terminal.integrated.profiles.osx` in [.vscode/settings.json](.vscode/settings.json) is used to add the workspace root folder to `$PATH` so that [dev](dev) script can be executed. 

The [.env](.env) file is to be created with the following OS env variables defined:

```text
export QUANT_CONNECT_USER_ID=?
export QUANT_CONNECT_API_TOKEN=?
export PATH="$PATH:$(pwd)"
```
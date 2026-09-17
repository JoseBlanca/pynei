# Create a pyodide testing environment

## Install pyodide

Download the latest pyodide release from: https://github.com/pyodide/pyodide/releases

Since pyodide 314.0.0 the releases are numbered after the CPython version they
bundle, so the 314.x series ships Python 3.14.

For instance, you can download:
https://github.com/pyodide/pyodide/releases/download/314.0.7/pyodide-314.0.7.tar.bz2

Unzip the pyodide distribution.

```
$ mkdir ~/.pyodide
$ mkdir ~/.pyodide/v314.0.7
$ tar -xvjf pyodide-314.0.7.tar.bz2 -C ~/.pyodide/v314.0.7/
```

## Install pyodide build

```
$ pip install pyodide-build
```

## Create the virtual enviroment

```
$ pyodide venv .venv-pyodide
Starting new HTTPS connection (1): raw.githubusercontent.com:443
Downloading Pyodide cross-build 
Installing Pyodide cross-build environment
Using Pyodide cross-build environment version: 314.0.7
Creating Pyodide virtualenv at .venv-pyodide
... Configuring virtualenv
... Installing standard library
Successfully created Pyodide virtual environment!
```

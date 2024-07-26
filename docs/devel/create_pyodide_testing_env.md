# Create a pyodide testing environment

## Install pyodide

Download the latest pyodide release from: https://github.com/pyodide/pyodide/releases

For instance, you can download: https://github.com/pyodide/pyodide/releases/download/0.26.1/pyodide-0.26.1.tar.bz2

Unzip the pyodide distribution.

```
$ mkdir ~/.pyodide
$ mkdir ~/.pyodide/v0.34.0
$ tar -xvjf pyodide-0.26.1.tar.bz2 -C ~/.pyodide/v0.34.0/
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
Using Pyodide cross-build environment version: 0.26.1
Creating Pyodide virtualenv at .venv-pyodide
... Configuring virtualenv
... Installing standard library
Successfully created Pyodide virtual environment!
```


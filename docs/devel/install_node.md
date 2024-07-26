# Install the nvm node manager

The documentation is: https://github.com/nvm-sh/nvm

$ wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

creo un script: activate_node

```
export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" # This loads nvm
```

Test the installation with:

command -v nvm

Now install the latest node:

```

$ nvm install node
$ nvm use node

```

Node should be available

```
$ node -v
v22.5.1
```
# The Rust spike

An experiment, not part of pynei: a crate with a VCF chunk parser, and z'z and
the symmetric eigendecomposition through faer, exposed with pyo3, to measure
Rust against the Python and numpy pynei has, natively and under pyodide. The
numbers are in issue #18.

## Native

Into the project venv, from the root of the project. `uv run --with maturin`
would install into uv's temporary overlay, not the venv:

    VIRTUAL_ENV=$PWD/.venv uvx maturin develop --release -m spike/pynei_spike/Cargo.toml

## pyodide

It needs a host Python that is not free threaded, pyodide-build, the
emscripten that the pyodide runtime was built with, 5.0.3 for pyodide
314.0.7, and the cross build env of that same version:

    uv venv --python cpython-3.14.5-macos-aarch64-none venv314
    VIRTUAL_ENV=$PWD/venv314 uv pip install pyodide-build
    git clone --depth 1 https://github.com/emscripten-core/emsdk.git
    (cd emsdk && ./emsdk install 5.0.3 && ./emsdk activate 5.0.3)
    source emsdk/emsdk_env.sh
    rustup target add wasm32-unknown-emscripten
    venv314/bin/pyodide xbuildenv install 314.0.7
    cd spike/pynei_spike && pyodide build

The wheel lands in dist/ with the pyemscripten ABI tag, which is the one
micropip accepts, `micropip.install("emfs:/path/to/the.whl")`.

rayon does not exist in wasm, there are no threads, and faer's rayon feature
does not build for emscripten, so both are pulled in only off the wasm
targets, `cfg(not(target_family = "wasm"))` in Cargo.toml and lib.rs.

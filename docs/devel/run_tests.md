podman run --rm  pynei_testing uv run --python cpython-3.13.3+freethreaded pytest test/
podman run --rm  pynei_testing uv run pytest test/
podman run --rm  pynei_testing uv run pytest --runtime node --run-in-pyodide test/

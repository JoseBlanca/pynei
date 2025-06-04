from subprocess import run
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
HOME_DIR = Path.home()


def run_command_in_container(cmd_to_run_in_container: list[str]):
    volume_mounts = [
        f"{PROJECT_DIR}/src:/app/src:ro",
        f"{PROJECT_DIR}/test:/app/test:ro",
        f"{PROJECT_DIR}/pyproject.toml:/app/pyproject.toml:ro",
        f"{PROJECT_DIR}/uv.lock:/app/uv.lock:ro",
        f"{PROJECT_DIR}/.python-version:/app/.python-version:ro",
        f"{PROJECT_DIR}/pytest.ini:/app/pytest.ini:ro",
        f"{PROJECT_DIR}/README.md:/app/README.md:ro",
        f"{HOME_DIR}/.cache/uv:/root/.cache/uv",
    ]
    cmd = [
        "podman",
        "run",
        "--rm",
    ]
    for mount in volume_mounts:
        cmd.extend(["-v", mount])

    cmd.extend(
        [
            "-w",
            "/app",
            "pynei_testing",
        ]
    )
    cmd.extend(cmd_to_run_in_container)

    run(cmd)


STD_TEST = [
    "uv",
    "run",
    "--link-mode=copy",
    "pytest",
    "test/",
]
PYODIDE_TEST = ["uv", "run", "pytest", "--runtime", "node", "--run-in-pyodide", "test/"]
FREE_THREADED_TEST = [
    "uv",
    "run",
    "--python",
    "cpython-3.13.3+freethreaded",
    "pytest",
    "test/",
]
run_command_in_container(STD_TEST)
run_command_in_container(PYODIDE_TEST)

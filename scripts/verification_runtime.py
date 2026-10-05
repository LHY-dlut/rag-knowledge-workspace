"""Keep isolated acceptance runtimes for diagnostics, with owned process cleanup."""

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


@contextmanager
def isolated_runtime(prefix):
    root = Path(__file__).resolve().parent.parent / ".acceptance" / "runtime"
    directory = root / f"{prefix}{uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    yield str(directory)


def stop_process(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()

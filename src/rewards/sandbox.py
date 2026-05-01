"""Safe code execution sandbox for reward computation."""

import subprocess
import sys
import tempfile
import os
from dataclasses import dataclass


@dataclass
class SandboxResult:
    """Result of sandboxed code execution."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    return_code: int = -1


# Imports that are blocked in the sandbox for safety
BLOCKED_IMPORTS_HOOK = '''
import builtins
_original_import = builtins.__import__

_BLOCKED_MODULES = {
    'subprocess', 'shutil', 'socket', 'requests', 'urllib',
    'http', 'ftplib', 'smtplib', 'ctypes', 'multiprocessing',
    'signal', 'importlib', 'pickle', 'shelve',
}

def _safe_import(name, *args, **kwargs):
    top_level = name.split('.')[0]
    if top_level in _BLOCKED_MODULES:
        raise ImportError(f"Import of '{name}' is not allowed in sandbox")
    return _original_import(name, *args, **kwargs)

builtins.__import__ = _safe_import
'''


def sandbox_execute(
    code: str,
    timeout: int = 10,
    memory_limit_mb: int = 512,
) -> SandboxResult:
    """
    Execute Python code in a sandboxed subprocess.

    Protection layers:
    1. Subprocess isolation (separate process)
    2. Timeout enforcement
    3. Blocked dangerous imports
    4. Memory limit (best-effort via resource limits on Linux)

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds
        memory_limit_mb: Maximum memory in MB (Linux only)

    Returns:
        SandboxResult with execution outcome
    """
    # Prepend import blocking hook and resource limits
    resource_limit = ""
    if sys.platform != "win32":
        resource_limit = f"""
import resource
try:
    resource.setrlimit(resource.RLIMIT_AS, ({memory_limit_mb * 1024 * 1024}, {memory_limit_mb * 1024 * 1024}))
except (ValueError, resource.error):
    pass
"""

    full_code = BLOCKED_IMPORTS_HOOK + resource_limit + "\n" + code

    # Write to temp file and execute
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(full_code)
        temp_path = f.name

    try:
        # Run in isolated subprocess
        env = os.environ.copy()
        # Remove potentially dangerous env vars
        for key in ["PYTHONSTARTUP", "PYTHONPATH"]:
            env.pop(key, None)

        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=tempfile.gettempdir(),
        )

        return SandboxResult(
            success=(result.returncode == 0),
            stdout=result.stdout[:10000],  # cap output size
            stderr=result.stderr[:10000],
            timed_out=False,
            return_code=result.returncode,
        )

    except subprocess.TimeoutExpired:
        return SandboxResult(
            success=False,
            stderr="Execution timed out",
            timed_out=True,
        )
    except Exception as e:
        return SandboxResult(
            success=False,
            stderr=str(e),
        )
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def execute_with_tests(
    code: str,
    test_code: str,
    entry_point: str,
    timeout: int = 10,
) -> SandboxResult:
    """
    Execute generated code with its test harness.

    Combines the generated function code with the test code and
    runs the check function.
    """
    full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"
    return sandbox_execute(full_code, timeout=timeout)

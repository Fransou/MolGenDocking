import subprocess
import time
from typing import Generator, Literal

import pytest
import requests

# Define allowed accelerator types
AcceleratorType = Literal["cpu", "gpu"]


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom pytest CLI options."""
    parser.addoption(
        "--accelerator",
        action="store",
        default="cpu",
        choices=["cpu", "gpu"],
        help="Select the hardware accelerator to use for tests (cpu or gpu).",
    )
    parser.addoption(
        "--skip-docking",
        action="store_true",
        default=True,
        help="Skip docking tests when set.",
    )
    parser.addoption(
        "--include-docking",
        dest="skip_docking",
        action="store_false",
    )
    parser.addoption(
        "--start-server",
        action="store_true",
        default=False,
        help="Start the uvicorn server before running tests and stop it after.",
    )


# =============================================================================
# Server Management
# =============================================================================

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5001
SERVER_STARTUP_TIMEOUT = 30  # seconds
SERVER_HEALTH_CHECK_INTERVAL = 0.5  # seconds


def _wait_for_server_ready(host: str, port: int, timeout: float) -> bool:
    """
    Wait for the server to become ready.

    Args:
        host: Server host address.
        port: Server port.
        timeout: Maximum time to wait in seconds.

    Returns:
        True if server is ready, False if timeout exceeded.
    """
    start_time = time.time()
    url = f"http://{host}:{port}/liveness"

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(SERVER_HEALTH_CHECK_INTERVAL)

    return False


def _is_server_running(host: str, port: int) -> bool:
    """
    Check if the server is already running.

    Args:
        host: Server host address.
        port: Server port.

    Returns:
        True if server is running, False otherwise.
    """
    try:
        response = requests.get(f"http://{host}:{port}/liveness", timeout=1)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@pytest.fixture(scope="session")
def uvicorn_server(
    request: pytest.FixtureRequest,
) -> Generator[subprocess.Popen | None, None, None]:
    """
    Fixture to start and stop the uvicorn server.

    This fixture starts the uvicorn server before tests run and stops it after
    all tests are complete. It only starts the server if --start-server is passed.

    Yields:
        The subprocess.Popen object for the server, or None if not started.
    """
    start_server = request.config.getoption("--start-server")

    if not start_server:
        yield None
        return

    # Check if server is already running
    if _is_server_running(SERVER_HOST, SERVER_PORT):
        print(f"\nServer already running on {SERVER_HOST}:{SERVER_PORT}")
        yield None
        return

    # Start the uvicorn server
    print(f"\nStarting uvicorn server on {SERVER_HOST}:{SERVER_PORT}...")
    process = subprocess.Popen(
        [
            "buffer_time=1uvicorn",
            "--host",
            SERVER_HOST,
            "--port",
            str(SERVER_PORT),
            "mol_gen_docking.server:app",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    if _wait_for_server_ready(SERVER_HOST, SERVER_PORT, SERVER_STARTUP_TIMEOUT):
        print(f"Server started successfully (PID: {process.pid})")
    else:
        process.terminate()
        process.wait()
        pytest.fail(f"Server failed to start within {SERVER_STARTUP_TIMEOUT} seconds")

    yield process

    # Cleanup: stop the server
    print(f"\nStopping uvicorn server (PID: {process.pid})...")
    process.terminate()
    try:
        process.wait(timeout=10)
        print("Server stopped successfully")
    except subprocess.TimeoutExpired:
        print("Server did not stop gracefully, killing...")
        process.kill()
        process.wait()


@pytest.fixture(scope="session")
def server_url(uvicorn_server: subprocess.Popen | None) -> str:
    """
    Fixture that returns the server URL.

    This fixture depends on uvicorn_server to ensure the server is running
    when --start-server is passed.

    Returns:
        The server URL string.
    """
    return f"http://{SERVER_HOST}:{SERVER_PORT}"


# =============================================================================
# Accelerator Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def accelerator(request: pytest.FixtureRequest) -> AcceleratorType:
    """
    Pytest fixture returning the selected accelerator.

    Example:
        def test_example(accelerator: AcceleratorType) -> None:
            if accelerator == "gpu":
                ...
    """
    accel: str = request.config.getoption("--accelerator")
    if bool(request.config.getoption("--skip-docking")):
        pytest.skip("Skipping docking tests")
    assert accel in ("cpu", "gpu")
    return accel


@pytest.fixture(scope="session")
def has_gpu(accelerator: AcceleratorType) -> bool:
    """
    Convenience fixture: True if accelerator == 'gpu'.
    """
    return accelerator == "gpu"

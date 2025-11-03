from typing import Literal

import pytest

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
        default=False,
        help="Skip docking tests when set.",
    )


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

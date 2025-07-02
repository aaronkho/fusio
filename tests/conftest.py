import pytest
from pathlib import Path


@pytest.fixture(scope='session')
def gacode_file_path():
    return Path(__file__).parent / 'data' / 'test_input.gacode'

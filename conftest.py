import pytest

@pytest.fixture
def model_name():
    return "LR"

@pytest.fixture
def file_path():
    return "test.csv"
import pytest
import numpy as np
import xarray as xr
from fusio.classes.io import io
from fusio.classes.gacode import gacode_io


@pytest.mark.usefixtures('gacode_file_path')
class TestInitialization():

    def test_empty_class_creation(self):
        assert isinstance(gacode_io(), gacode_io)

    def test_initialized_input_class_creation(self, gacode_file_path):
        g = gacode_io(input=gacode_file_path)
        assert isinstance(g, gacode_io)
        assert g.has_input

    def test_initialized_output_class_creation(self, gacode_file_path):
        g = gacode_io(output=gacode_file_path)
        assert isinstance(g, gacode_io)
        assert g.has_output

    def test_read_native_input(self, gacode_file_path):
        g = gacode_io()
        g.read(gacode_file_path, side='input')
        assert g.has_input

    def test_read_native_output(self, gacode_file_path):
        g = gacode_io()
        g.read(gacode_file_path, side='output')
        assert g.has_output

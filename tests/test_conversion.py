import os
import pathlib
import unittest

import numpy as np
import pytest
from igor2 import binarywave

from hystorian.io import hyFile, utils

filepath = pathlib.Path("tests/test_files/test.hdf5")


@pytest.fixture(autouse=True)
def teardown_module():
    if filepath.is_file():
        os.remove(filepath)


class TestHyFileConversion:
    def test_extraction_ibw(self):
        path = pathlib.Path("tests/test_files/raw_files/test_ibw.ibw")
        with hyFile.HyFile(filepath, "r+") as f:
            f.extract_data(path, ignore_if_exist=False)

            tmpdata = binarywave.load(path)["wave"]
            metadata = {}

            for meta in tmpdata["note"].decode("ISO-8859-1").split("\r")[:-1]:
                if len(meta.split(":")) == 2:
                    metadata[meta.split(":")[0]] = meta.split(":")[1]

            # for key in f.read(f"metadata/{path.stem}"):
            #    assert key in list(metadata.keys())

    def test_extraction_gsf(self):
        path = pathlib.Path("tests/test_files/raw_files/test_gsf.gsf")
        with hyFile.HyFile(filepath, "r+") as f:
            f.extract_data(path, ignore_if_exist=False)

            gsfFile = open(path, "rb")  # + ".gsf", "rb")

            metadata = {}

            # check if header is OK
            if not (gsfFile.readline().decode("UTF-8") == "Gwyddion Simple Field 1.0\n"):
                gsfFile.close()
                raise ValueError("File has wrong header")

            term = b"00"
            # read metadata header
            while term != b"\x00":
                line_string = gsfFile.readline().decode("UTF-8")
                metadata[line_string.rpartition("=")[0].strip()] = line_string.rpartition("=")[2].strip()
                term = gsfFile.read(1)

                gsfFile.seek(-1, 1)

            gsfFile.close()

            assert f[f"metadata/{path.stem}"].attrs["XReal"] == float(metadata["XReal"])
    
    @pytest.mark.skip(reason="The file is not available in the repository.")
    def test_extraction_ardf_sspfm(self):
        path = pathlib.Path("tests/test_files/raw_files/test_sspfm_ardf.ARDF")
        with hyFile.HyFile(filepath, "r+") as f:
            f.extract_data(path, ignore_if_exist=False)
            channels = set(f.read("datasets/test_sspfm_ardf/"))
            value = f.read("datasets/test_sspfm_ardf/Amp/retrace")[0][0][0]

        assert channels == {
            "Amp",
            "Bias",
            "MapAdhesion",
            "MapHeight",
            "Phase",
            "Freq",
            "Raw",
            "ZSnsr",
            "Defl",
            "Phas2",
        }

        # assert pytest.approx(value) == 757608337.0
        
    @pytest.mark.skip(reason="The file is not available in the repository.")
    def test_extraction_ardf_fmap(self):
        path = pathlib.Path("tests/test_files/raw_files/test_fmap_ardf.ARDF")
        with hyFile.HyFile(filepath, "r+") as f:
            f.extract_data(path, ignore_if_exist=False)
            channels = set(f.read("datasets/test_fmap_ardf/"))
            value = f.read("datasets/test_fmap_ardf/Bias/trace")[0][0][0]

        assert channels == {"Bias", "MapAdhesion", "Raw", "MapHeight", "ZSnsr", "Defl"}
        # assert pytest.approx(value) == 2147483647.0

    def test_nanonis(self):
        path = pathlib.Path("tests/test_files/raw_files/test_nanonis.000")
        with hyFile.HyFile(filepath, "r+") as f:
            f.extract_data(path, ignore_if_exist=False)
            channels = set(f.read("datasets/test_nanonis"))
            value = f.read("datasets/test_nanonis/HeightRetrace")[0][0]

        assert channels == {
            "AmplitudeErrorTrace",
            "AmplitudeRetrace",
            "HeightRetrace",
            "PhaseTrace",
            "ZSensorRetrace",
        }
        assert value == 29004


if __name__ == "__main__":
    unittest.main()

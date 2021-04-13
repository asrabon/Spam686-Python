# Standard Python Libraries
import os
import shutil
import sys

# 3rd Party Python Libraries
from PIL import Image
import numpy as np
from oct2py import octave
from PIL import Image

# Local Python Imports
from spam686 import spam_extract

octave.addpath('./test/')

# Constants
TEST_IMAGE_PATH = os.path.abspath(os.path.join("./", "test", "test.png"))


def test_spam_features():
    # Python Feature Extraction
    img = Image.open(TEST_IMAGE_PATH)
    img_arr = np.asarray(img, np.double)
    python_spam_features = spam_extract(img_arr, 3).flatten()

    # Octave/Matlab Feature Extraction
    octave_spam_features = octave.spam686(TEST_IMAGE_PATH).flatten()

    assert np.allclose(octave_spam_features, python_spam_features)

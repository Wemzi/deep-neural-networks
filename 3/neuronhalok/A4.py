# Download tester
!rm tester.py
!wget http://nipg12.inf.elte.hu/~vavsaai@nipg.lab/annbsc21_p1_hw2/tester.py 

import numpy as np

%tensorflow_version 2.x
import tensorflow as tf
import tensorflow.keras.models
import tensorflow.keras.optimizers
import tensorflow.keras.layers
import tensorflow.keras.activations
import tensorflow.keras.callbacks

import matplotlib.pyplot as plt

# import tester after importing tensorflow, to make sure correct tf version is imported
from tester import Tester

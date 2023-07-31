# CNN-LSTM Model for testing on gathered human activity data.

# Using pytorch implementation.

# Import statements
import torch
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylabimport rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim

# Matplotlib additional arguments.
%matplotlib inline
%config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
rcParams['figure.figsize'] = 14, 10
register_matplotlib_converters()
RANDOM_SEED = 33
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


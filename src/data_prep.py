#!/usr/bin/env python3

# ---------------------------------------------------------------------------- #
#                                                                              #
# CATHODE ~ Putting (A)NODEs to work                                           #
#          (for financial time-series prediction and simulation)               #
#                                                                              #
# |> Data download preparation and pre-processing <|                           #
#                                                                              #
# (C) 2020-* Emanuele Ballarin <emanuele@ballarin.cc>                          #
# (C) 2020-* Arianna Tasciotti                                                 #
# (C) 2020-* Milton Nicolas Plasencia Palacios                                 #
#                                                                              #
# Distribution: Apache License 2.0                                             #
#                                                                              #
# Eventually-updated version: https://github.com/emaballarin/cathode           #
# Restricted-access material: https://bit.ly/cathode-sml-reserved              #
#                                                                              #
# ---------------------------------------------------------------------------- #


# ------- #
# IMPORTS #
# ------- #

import numpy            # Just force the right OpenMP implementation!

import pandas as pd     # Dataframe provider for CSV
import vaex as vx       # Dataframe provider for HDF5

import os               # Access to files



# ----------- #
# DATA IMPORT #
# ----------- #


# Read dataset from file
dataset = pd.read_csv(os.path.join("..", "data", "WIKI_PRICES_QUANDL.csv"))



# --------------------------- #
# DATA CLEANING / PREPARATION #
# --------------------------- #


# Remove data that are deterministic functions / deterministic functions of each other
dataset.drop(["ex-dividend", "split_ratio", "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"], axis = 1, inplace = True)

# Remove rows with missing data, since they are present in minimal amount
dataset.dropna(inplace=True)

# Convert string-based dates to integers (days) since first date in chronological order (day 0)
dataset['date'] = pd.to_datetime(dataset['date'])
basedate = dataset.date.min()
dataset['date'] = (dataset['date'] - basedate).dt.days

# Make all numeric data to be float64 to ease computation
dataset['date'] = dataset['date'].astype('float64')



# ----------- #
# DATA EXPORT #
# ----------- #


# Vaex-ify Pandas dataset and export to little-endian HDF5
vaex_dataset = vx.from_pandas(dataset)
vaex_dataset.export_hdf5("../data/WIKI_PRICES_QUANDL.hdf5", column_names = None, byteorder = '<', progress = False)

# Greet user!
print('Success!')

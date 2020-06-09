#!/usr/bin/env python3

# ---------------------------------------------------------------------------- #
#                                                                              #
# CATHODE ~ Putting (A)NODEs to work                                           #
#          (for financial time-series prediction and simulation)               #
#                                                                              #
# |> Plotting convenience functions <|                                         #
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

import numpy        # Just force the right OpenMP implementation!

import vaex         # Dataframe provider
import vaex as vx   # Dataframe provider (shorthand)

import matplotlib.pyplot as plt    # Basic plotting

# Self-rolled utilities
from src.util.datamanip import data_by_tick
from src.util.datamanip import data_by_tick_col



# --------- #
# FUNCTIONS #
# --------- #


# Plot a time series with piecewise linear interpolants (by tick, column)
def data_timeplot(data, tickname, coln):
    assert(coln != 'date')
    to_be_plotted = data_by_tick_col(data, tickname, ('date', coln))
    plt.plot(to_be_plotted['date'].values, to_be_plotted[coln].values)

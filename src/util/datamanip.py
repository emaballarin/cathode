#!/usr/bin/env python3

# ---------------------------------------------------------------------------- #
#                                                                              #
# CATHODE ~ Putting (A)NODEs to work                                           #
#          (for financial time-series prediction and simulation)               #
#                                                                              #
# |> Data manipulation convenience functions <|                                #
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

import numpy  # Just force the right OpenMP implementation!

import vaex  # Dataframe provider


# --------- #
# FUNCTIONS #
# --------- #


# Filter data by ticker
def data_by_tick(data, tickname):
    return data[data["ticker"] == tickname]


# Filter data by ticker and column name
def data_by_tick_col(data, tickname, coln):
    return data_by_tick(data, tickname)[coln]

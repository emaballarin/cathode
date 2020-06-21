#!/usr/bin/env python3

# ---------------------------------------------------------------------------- #
#                                                                              #
# CATHODE ~ Putting (A)NODEs to work                                           #
#          (for financial time-series prediction and simulation)               #
#                                                                              #
# |> Datasets/Data-loading convenience functions <|                            #
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
import vaex as vx

import matplotlib.pyplot as plt  # Basic plotting

# Self-rolled utilities
from src.util.datamanip import data_by_tick
from src.util.datamanip import data_by_tick_col

# Neural Networks / Neural ODEs
import torch as th
from torch.utils.data import Dataset


# --------- #
# FUNCTIONS #
# --------- #


# DESC
# XYZ


# ------- #
# CLASSES #
# ------- #


# PyTorch dataset scaffold for QUANDL-like stock data (HDF5)
class StockDataset(Dataset):
    def __init__(
        self,
        hdf5_file,
        company,
        col_n,
        split_ratio,
        window_size,
        sliding_step=1,
        batch_size=1,
        normalize=True,
        train=True,
    ):
        self.dataframe = vx.open(hdf5_file)
        self.split_ratio = split_ratio
        self.window_size = window_size
        self.sliding_step = sliding_step
        self.train = train
        self.batch_size = batch_size

        if isinstance(col_n, tuple):
            col_n = tuple(["date"]) + col_n
        else:
            col_n = tuple(["date"]) + tuple([col_n])

        self.full_data = data_by_tick_col(self.dataframe, company, col_n)

        if normalize:
            for col in col_n:
                if col != "date":
                    self.full_data[col] = (
                        self.full_data[col] - self.full_data[col].mean()
                    ) / self.full_data[col].std()

        self.full_data = (self.full_data).values

        self.data = self.full_data[
            : int(len(self.full_data) * self.split_ratio)
        ]  # training set

        assert len(self.data) >= self.window_size
        assert (
            len(self.data) - self.window_size + 1
        ) % self.sliding_step == 0  # self.data is training data

        if not train:
            self.sliding_step = self.window_size
            self.test_data = self.full_data[int(len(self.data)) :]

        self.offset = (
            int((len(self.data) - self.window_size + 1) / self.sliding_step)
            % self.batch_size
        )

        self.nr_of_batches = (
            int((len(self.data) - self.window_size + 1) / self.sliding_step)
            - self.offset
        ) / self.batch_size

    def __len__(self):
        if self.train:
            return (
                int((len(self.data) - self.window_size + 1) / self.sliding_step)
                - self.offset
            )
        else:
            assert self.batch_size == 1
            return (
                1
                + int((len(self.data) - self.window_size + 1) / self.sliding_step)
                - self.offset
            )

    def __getitem__(self, idx: int):

        idx = ((int(idx) % int(self.batch_size)) * int(self.nr_of_batches)) + int(
            idx
        ) // int(self.batch_size)

        idx = idx + self.offset

        if self.train:
            window = self.data[
                int(self.sliding_step * idx) : int(
                    (self.sliding_step * idx) + self.window_size
                )
            ]
            input_window, output_window = (
                window[: int(len(window) * self.split_ratio)],
                window[int(len(window) * self.split_ratio) :],
            )
        else:
            l = int((len(self.data) - self.window_size + 1) / self.sliding_step)

            if idx == l:
                input_window, output_window = (
                    [],
                    self.test_data,
                )

            else:
                window = self.data[
                    int(self.sliding_step * idx) : int(
                        (self.sliding_step * idx) + self.window_size
                    )
                ]

                input_window, output_window = (
                    window[: int(len(window) * self.split_ratio)],
                    [],
                )

        out_dict = {"past": input_window, "future": output_window}
        return out_dict

#!/usr/bin/env bash

# ---------------------------------------------------------------------------- #
#                                                                              #
# CATHODE ~ Putting (A)NODEs to work                                           #
#          (for financial time-series prediction and simulation)               #
#                                                                              #
# |> Data download (manual) and unpacking script <|                            #
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

# The data required to perform the experiments and subsequent analysis, which this
# script operates on are provided by Quandl (https://www.quandl.com/) and must be
# downloaded manually, after registration, from here:
# https://www.quandl.com/tables/WIKIP/WIKI-PRICES/export
#
# The resulting *.zip file has a name in the form WIKI_PRICES_{CODE}.zip, where the
# {CODE} part is just a string of alphanumeric elements.
#
# For the sake of a practical execution, such string must be specified below,
# replacing the underscore.

#PERSONAL_CODE="_"
PERSONAL_CODE="212b326a081eacca455e13140d7bb9db"

# The resulting *.zip file must be put inside the ../data/ directory (or a symlink
# to it). Afterwards this script can be executed.

################################################################################

PRE_CALLDIR="$(pwd)"

cd "../data/"

COMPOSITE_FILENAME_ZIP="WIKI_PRICES_$PERSONAL_CODE.zip"
COMPOSITE_FILELOCT_ZIP="./$COMPOSITE_FILENAME_ZIP"

COMPOSITE_FILENAME_CSV="WIKI_PRICES_$PERSONAL_CODE.csv"
COMPOSITE_FILELOCT_CSV="./$COMPOSITE_FILENAME_CSV"

unzip "$COMPOSITE_FILELOCT_ZIP"
sleep 2
rm -f "$COMPOSITE_FILELOCT_ZIP"

mv "$COMPOSITE_FILELOCT_CSV" "WIKI_PRICES_QUANDL.csv"
sleep 1

cd "$PRE_CALLDIR"

echo "Operation successful!"

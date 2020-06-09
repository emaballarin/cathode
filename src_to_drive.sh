#!/usr/bin/zsh

PREV_CALLDIR="$(pwd)"

#####################

# Set directory
cd "/home/emaballarin/DSSC/smlearning/cathode/"

# Remove precompiled Python files
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

# Let the magic happen ;)
rclone copyto "./src" "drive_ballarin:/cathode-sml/src"

#####################

cd "$PREV_CALLDIR"

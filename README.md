# Acoustic Analysis Toolkit (alts.py)

## Overview

The alts.py script is a comprehensive toolkit designed for acoustic data analysis. It provides functionality for calculating pulse intervals, filtering data based on specific conditions, and visualizing sound pressure level (SPL) distributions and cross-correlations. This toolkit is intended for researchers and analysts working with acoustic datasets, particularly those involving marine environments.

## Features
•	Pulse Interval Calculation: Computes the pulse interval between consecutive acoustic events.
•	Mean Angle Calculation: Estimates the mean angle of pulses within a time bin.
•	Filtering Conditions: Offers multiple filtering options for noise conditions, from raw data viewing to advanced parameter settings for noisy environments.
•	Data Visualization: Generates plots for SPL time distribution, SPL histograms, and cross-correlation between different SPL measurements.

## Installation

To use the alts.py script, you need to have Python installed along with the following libraries:
•	pandas
•	numpy
•	matplotlib
•	seaborn
•	scipy

## Usage

## Pulse Interval Calculation

To calculate the pulse interval between consecutive acoustic events, use the calculate_pulse_interval function:
'''
from alts import alts_filter

# Assuming df is your DataFrame containing a 'datetime' column
df = alts_filter.calculate_pulse_interval(df)
'''

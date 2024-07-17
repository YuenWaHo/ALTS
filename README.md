# Acoustic Analysis Toolkit (alts.py)

## Overview

The `alts.py` script is a comprehensive toolkit designed for acoustic data analysis. It provides functionality for calculating pulse intervals, filtering data based on specific conditions, and visualizing sound pressure level (SPL) distributions and cross-correlations. This toolkit is intended for researchers and analysts working with acoustic datasets, particularly those involving marine environments.

## Features

- **Pulse Interval Calculation**: Computes the pulse interval between consecutive acoustic events.
- **Mean Angle Calculation**: Estimates the mean angle of pulses within a time bin.
- **Filtering Conditions**: Offers multiple filtering options for noise conditions, from raw data viewing to advanced parameter settings for noisy environments.
- **Data Visualization**: Generates plots for SPL time distribution, SPL histograms, and cross-correlation between different SPL measurements.

## Installation

To use the `alts.py` script, you need to have Python installed along with the following libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scipy

## Usage

### Pulse Interval Calculation
To calculate the pulse interval between consecutive acoustic events, use the `calculate_pulse_interval` function:

```python
from alts import alts_filter

# Assuming df is your DataFrame containing a 'datetime' column
df = alts_filter.calculate_pulse_interval(df)
```

### Filtering Conditions
You can filter your data based on different noise conditions:
```python
# For raw viewer without any filter
alts_filter.filter_condition(noise_condition='1')

# For mild filter, recommended for most cases
alts_filter.filter_condition(noise_condition='2')

# For very noisy conditions
alts_filter.filter_condition(noise_condition='3')

# Advanced parameter settings
alts_filter.filter_condition(noise_condition='4')
```
### Data Visualization
The toolkit includes several functions for visualizing SPL data:
- SPL Time Distribution Plot
```python
alts_filter.spl_time_distribution_plot(df)
```
- SPL Distribution Plot
```python
alts_filter.spl_distribution_plot(df)
```
- SPL Cross-Correlation Plot
```python
alts_filter.spl_cross_corr(df)
```

### Bokeh Visualization Plot
The toolkit includes a function for Bokeh visualization of acoustic data results:
#### `plot_alts_result`

This function visualizes acoustic data using Bokeh. The parameters are:
- `dff`: DataFrame containing the filtered data.
- `dfr`: DataFrame containing the raw data.
- `time_diff`: Time difference parameter.
- `atag`: Annotation tag for the plot.

Example usage:
```python
from alts import alts_visualization

# Assuming dff and dfr are your DataFrames containing filtered and raw data respectively
alts_visualization.plot_alts_result(dff=dff, dfr=dfr, time_diff=0, atag=2)
```












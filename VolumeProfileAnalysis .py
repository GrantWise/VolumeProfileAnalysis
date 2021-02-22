# -*- coding: utf-8 -*-
"""

Based on Medium Article https://medium.com/swlh/how-to-analyze-volume-profiles-with-python-3166bb10ff24

Created on Mon Feb 22 18:58:39 2021

@author: Grant
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

pio.renderers.default = 'browser'

df = pd.read_json('C:\\ \\Data\BTC_USDT-1h.json')
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
convert_df = pd.to_datetime(df['date'], unit='ms')
df['date'] = convert_df
df_recent = df.iloc[-1250:]
df_recent.head()

def get_dist_plot(c, v, kx, ky):
    fig = go.Figure()
    fig.add_trace(go.Histogram(name='Vol Profile', x=c, y=v, nbinsx=150, 
                               histfunc='sum', histnorm='probability density',
                               marker_color='#B0C4DE'))
    fig.add_trace(go.Scatter(name='KDE', x=kx, y=ky, mode='lines', marker_color='#D2691E'))
    return fig

px.line(df_recent, x='date', y='close').show()

# Set Marker Size in Plot
pk_marker_args=dict(size=10)

# A kernel density estimator (KDE) is a non-parametric way to estimate the probability density function (PDF) of a random variable.
# This allows us to represent our distribution as a smooth and continuous curve.
# Num_samples is the number of bins in the histogram
kde_factor = 0.07
num_samples = 500
kde = stats.gaussian_kde(df_recent['close'],weights=df_recent['volume'],bw_method=kde_factor)
xr = np.linspace(df_recent['close'].min(),df_recent['close'].max(),num_samples)
kdy = kde(xr)
ticks_per_sample = (xr.max() - xr.min()) / num_samples


# Find the peaks
peaks,_ = signal.find_peaks(kdy)
pkx = xr[peaks]
pky = kdy[peaks]


# Find Prominant Peaks. Larger value identifies larger peaks only i.e. fewer, smaller value shows more smaller peaks
min_prom = 0.000005
peaks, peak_props = signal.find_peaks(kdy, prominence=min_prom)
pkx = xr[peaks]
pky = kdy[peaks]

fig = get_dist_plot(df_recent['close'], df_recent['volume'], xr, kdy)
fig.add_trace(go.Scatter(name='Peaks', x=pkx, y=pky, mode='markers', marker=pk_marker_args))

# Draw prominence lines
left_base = peak_props['left_bases']
right_base = peak_props['right_bases']
line_x = pkx
line_y0 = pky
line_y1 = pky - peak_props['prominences']

for x, y0, y1 in zip(line_x, line_y0, line_y1):
    fig.add_shape(type='line',
        xref='x', yref='y',
        x0=x, y0=y0, x1=x, y1=y1,
        line=dict(
            color='red',
            width=2,
        )
    )
fig.show()

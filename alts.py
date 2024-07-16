import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import datetime as dt
import bokeh
from bokeh.layouts import column, gridplot, row
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import ColumnDataSource, RangeTool, Span, HoverTool, Slider, DataTable, TableColumn, SelectEditor, NumberFormatter, Div, CustomJS
from bokeh.io import show
pd.set_option('display.float_format', '{:.5f}'.format)

class check_version:
    def print_requirement():
        # Installation instructions for the colleague
        print("\n#########  \nPlease ensure you have the following versions installed:")
        print("Pandas version: 1.5.2")
        print("NumPy version: 1.24.1")
        print("tkinter version: 8.5")
        print("Matplotlib version: 3.6.2")
        print("Seaborn version: 0.12.1")
        print("Bokeh version: 3.0.3")

        print("\nTo install these versions, you can use the following commands:")
        print("pip install pandas==1.5.2")
        print("pip install numpy==1.24.1")
        print("pip install tkinter==8.5")
        print("pip install matplotlib==3.6.2")
        print("pip install seaborn==0.12.1")
        print("pip install bokeh==3.0.3")


    def print_versions():
        import sys
        print('\n#########  \n')
        print('Python Version:', sys.version)
        print("\nPandas version:", pd.__version__)
        print("NumPy version:", np.__version__)
        print("tkinter version:", tk.TkVersion)
        print("Seaborn version:", sns.__version__)
        print("Bokeh version:", bokeh.__version__)

class alts_load:
    """
    A class for loading, processing, and analyzing acoustic data from CSV files.
    """

    @staticmethod
    def adjust_spl(dff):
        """
        Adjusts the Sound Pressure Level (SPL) values in the DataFrame.

        Parameters:
        dff (DataFrame): The DataFrame containing SPL1 and SPL2 columns to adjust.

        Returns:
        DataFrame: The adjusted DataFrame with modified SPL values.
        """
        dff.SPL1 = dff.SPL1 * 0.771
        dff.SPL2 = dff.SPL2 * 0.771
        return dff
    
    @staticmethod
    def alts_analysis(atag=2):
        """
        Loads acoustic data, adjusts SPL values, and calculates pulse intervals.

        Parameters:
        atag (int, optional): Determines whether to process front (2) or another position. Defaults to 2.

        Returns:
        DataFrame, or tuple of DataFrames: Processed DataFrame(s) with adjusted SPL values and pulse intervals.
        """
        dff = alts_load.load_csv()
        if dff is None:
            return None  # Early return if file loading is cancelled or fails

        dff = alts_load.adjust_spl(dff)
        print("#------         FRONT data loading completed    ------#")
        dff = alts_load.pulse_interval(dff)

        if atag == 2:
            dfr = alts_load.load_csv()
            if dfr is None:
                return dff  # Return only the front data if rear data loading is cancelled or fails

            dfr = alts_load.adjust_spl(dfr)
            print("#------         REAR data loading completed     ------#")
            dfr = alts_load.pulse_interval(dfr)
            print('#------ FRONT and REAR Pulse Interval Completed ------#')
            return dff, dfr
        else:
            print('#------ Pulse Interval Completed ------#')
            return dff

    @staticmethod
    def load_csv():
        """
        Opens a file dialog for the user to select a CSV file, then loads and processes the file.

        Returns:
        DataFrame: The processed DataFrame with added 'SPLR' column, or None if loading fails.
        """
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        if not file_path:
            print("File loading cancelled.")
            return None

        try:
            df = pd.read_csv(file_path)
            df = alts_load.process_dataframe(df)
            df['SPLR'] = df['SPL1'] / df['SPL2']
            return df
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    staticmethod
    def process_dataframe(df):
        """
        Processes a DataFrame by adjusting its structure based on the number of columns. 
        This method combines the logic for processing both four-column and five-column DataFrames.

        Parameters:
        df (DataFrame): The DataFrame to process.

        Returns:
        DataFrame: The processed DataFrame with a unified datetime column and, if applicable, a combined date and time.
        """
        if len(df.columns) == 4:
            # Process as four-column DataFrame
            df.columns = ['datetime', 'SPL1', 'SPL2', 'td']
            df['datetime'] = pd.to_datetime(df['datetime'], unit='s', origin='1904-01-01')
        elif len(df.columns) == 5:
            # Process as five-column DataFrame
            df.columns = ['date', 'time', 'SPL1', 'SPL2', 'td']
            df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
            df['time'] = pd.to_timedelta(df['time'].astype(float), unit='s')
            df['datetime'] = df['date'] + df['time']
        
        
        # Common processing for both DataFrame types
        df = alts_load.split_datetime(df)
        return df[['datetime', 'time', 'SPL1', 'SPL2', 'td']]
    
    @staticmethod
    def split_datetime(df):
        """
        Splits the 'datetime' column into separate date and time components.

        Parameters:
        df (DataFrame): The DataFrame to process.

        Returns:
        DataFrame: The DataFrame with a newly added 'time' column in seconds since midnight.
        """
        df['time'] = df['datetime'].dt.strftime('%H:%M:%S.%f')
        time_components = df['time'].str.split(':', expand=True)
        df['time'] = time_components[0].astype(int) * 3600 + time_components[1].astype(int) * 60 + time_components[2].astype(float)
        return df

    @staticmethod
    def pulse_interval(df):
        """
        Calculates the pulse interval between consecutive acoustic events.

        Parameters:
        df (DataFrame): The DataFrame to calculate intervals for.

        Returns:
        DataFrame: The DataFrame with an added 'PulseInterval' column representing the interval in seconds.
        """
        df = df.sort_values('datetime', ascending=True)
        df['datetime_next'] = df['datetime'].shift(-1)
        df['PulseInterval'] = (df['datetime_next'] - df['datetime']) / np.timedelta64(1, 's')
        return df

class alts_filter:
    def mean_angle(df): #Count number of pulses in time bin
        total_pulse = len(df)
        timebin = 1 #seconds
        n_box = (df.iloc[-1].datetime - df.iloc[0].datetime)/timebin #number of boxes

    @staticmethod
    def filter_condition(noise_condition='1'):
        SPLthreshold = 43
        parameters = {
            'tdswitch': 1,
            'Reflection': 0.0005,
            'SPLR': 0,
            'Isolation': 1.0,
            'smoothwin': 3,
            'smoothSPL': 1000000,
            'NpTrainMin': 5,
            'NpTrainMax': 1000000,
            'PiVar': 0.4,
            'tdVar': 1000000,
            'SPLDif': 1000000,
            'SPLVar': 1000000,
            'AvPiMax': 1000000,
            'MedPiMax': 1000000,
        }

        if noise_condition == '1':
            print("Raw viewer without any filter")
        elif noise_condition == '2':
            print("Recommended for most cases with a mild filter")
        elif noise_condition == '3':
            print("For very noisy conditions such as fixed monitoring close to shore with a lot of snapping shrimps")
            parameters.update({
                'Reflection': 0.0025,
                'Isolation': 0.2,
                'PiVar': 0.5,
                'tdVar': 0.5,
            })
        elif noise_condition == '4':
            print("Advanced parameter setting")
            parameters.update({
                'Reflection': 0.0025,
                'smoothwin': 1000000,
                'NpTrainMin': 6,
            })
        else:
            print('Unexpected Noise Condition')
            print("1: No filter applies. Just raw view") 
            print("2: Recommended for most cases with a mild filter") 
            print("3: For very noisy conditions such as fixed monitoring close to shore with a lot of snapping shrimps") 
            print("4: Advanced parameter setting") 

        return SPLthreshold, parameters

    @staticmethod
    def cleanup_td(df, tdswitch=1):
        ## 0th screening; delete td=0
        print('###--- 0th screening ---###')
        total_pulse = len(df)
        if tdswitch == 1:
            print("Exclude clicks without source direction information (td=+511 excluded)")
            df = df[(abs(df['td']) > 0) & (abs(df['td']) < 511)]
        elif tdswitch == 0:
            print('Include single trigger data without bearing angle information (include td=0)')
            df = df[df['td'] < 511]
        p = len(df)
        print("Total number of pulses: {}  ; remained after excluding time difference filter: {}".format(total_pulse, p))
        print("")
        return df

    @staticmethod
    def cleanup_lowint_reflect(df, SPLthreshold, Reflection):
        ## 1st screening; eliminate low intensity and reflections
        print('###--- 1st screening ---###')
        print('Select clicks with sound pressure >= {} counts. 1 count = 0.077 Pa approximately'.format(SPLthreshold))
        print('Eliminated possible reflections associated within {} ms after the previous pulse'.format(Reflection * 1000))
        total_pulse = len(df)
        df = df[(df['SPL1'] >= SPLthreshold) & (df['PulseInterval'] > Reflection)]
        p = len(df)
        print("Total number of pulses: {}  ; remained after threshold and reflection filter: {}".format(total_pulse, p))
        print("")
        return df

    @staticmethod
    def cleanup_iso_pulse(df, Isolation):
        ## 2nd screening; eliminate isolated pulse
        print('###--- 2nd screening ---###')
        print('Eliminated isolated pulses {} ms apart from both sides pulses'.format(Isolation * 1000))
        total_pulse = len(df)
        df['datetime_front'] = df['datetime'].shift(1)
        df['Pulse_time_diff_front'] = abs(df['datetime_front'] - df['datetime']) / np.timedelta64(1, 's')
        df['datetime_back'] = df['datetime'].shift(-1)
        df['Pulse_time_diff_back'] = abs(df['datetime_back'] - df['datetime']) / np.timedelta64(1, 's')
        df = df[(df['Pulse_time_diff_back'] <= Isolation) & (df['Pulse_time_diff_front'] <= Isolation)]
        p = len(df)
        print("Total number of pulses: {}  ; remained after isolated pulse filter: {}".format(total_pulse, p))
        print("")
        df = df[['datetime', 'SPL1', 'time', 'SPL2', 'td', 'PulseInterval', 'SPLR']]
        return df

    @staticmethod
    def cleanup_unusual_ipi(df, smoothwin):
        ## 3rd screening; eliminate unusual change of inter-pulse intervals
        print('###--- 3rd screening ---###')
        print("Select clicks with smoothly changed inter-pulse intervals with R=< {}".format(smoothwin))
        print("where R=(IPI present)/(IPI previous).   Accept  1/smoothIPI < R < smoothIPI")
        total_pulse = len(df)
        df['PI_next'] = df['PulseInterval'].shift(-1)
        df['R'] = df['PulseInterval'] / df['PI_next']
        df = df[(df['R'] <= smoothwin) & (df['R'] > (1 / smoothwin))]
        p = len(df)
        print("Total number of pulses: {}  ; remained after smoothing IPI: {}".format(total_pulse, p))
        print("")
        df = df[['datetime', 'SPL1', 'time', 'SPL2', 'td', 'PulseInterval', 'SPLR']]
        return df

    @staticmethod
    def cleanup_unusual_spl(df, smoothSPL):
        ## 4th screening; eliminate unusual change of SPL
        print('###--- 4th screening ---###')
        print("Select clicks with smoothly changed sound pressure with R=< {}".format(smoothSPL))
        print("where R=(SPL present)/(SPL previous).  Accept  1/smoothSPL < R < smoothSPL")
        total_pulse = len(df)
        df['SPL1_next'] = df['SPL1'].shift(-1)
        df['R'] = df['SPL1'] / df['SPL1_next']
        df = df[(df['R'] <= smoothSPL) & (df['R'] > 1 / smoothSPL)]
        p = len(df)
        print("Total number of pulses: {}  ; remained after smoothing SPL: {}".format(total_pulse, p))
        print("")
        df = df[['datetime', 'SPL1', 'time', 'SPL2', 'td', 'PulseInterval', 'SPLR']]
        return df

    @staticmethod
    def cleanup(df, noise_condition='1', pos='front'):
        SPLthreshold, parameters = alts_filter.filter_condition(noise_condition)
        
        print("----------PULSE BY PULSE FILTERING----------")
        if pos == 'rear':
            df = df[['datetime', 'SPL1B', 'SPL2B', 'tdB', 'PulseInterval', 'time', 'SPLR']]
            df.columns = ['datetime', 'SPL1', 'SPL2', 'td', 'PulseInterval', 'time', 'SPLR']
        else:
            df = df[['datetime', 'SPL1', 'SPL2', 'td', 'PulseInterval', 'time', 'SPLR']]
        
        original_number_click = len(df)
        ## 0th screening; delete td=0
        df = alts_filter.cleanup_td(df, tdswitch=parameters['tdswitch'])
        ## 1st screening; eliminate low intensity and reflections
        df = alts_filter.cleanup_lowint_reflect(df, SPLthreshold, parameters['Reflection'])
        ## 2nd screening; eliminate isolated pulse
        df = alts_filter.cleanup_iso_pulse(df, parameters['Isolation'])
        ## 3rd screening; eliminate unusual change of inter-pulse intervals
        df = alts_filter.cleanup_unusual_ipi(df, parameters['smoothwin'])
        ## 4th screening; eliminate unusual change of SPL
        df = alts_filter.cleanup_unusual_spl(df, parameters['smoothSPL'])
        
        post_processing_click = len(df)
        
        print(" ")
        print("Total number of pulses in the original data file = {} ".format(original_number_click))
        print('{} pulses remained after click-by-click filtering'.format(post_processing_click))
        print("Note that large numbers such as 1e*06 means the filtering function is disabled")
        print(" ")
        return df


    def pulse_train(df):
        Tpi = 200 ## Definition of the boundary of a pulse train.
        # Tpi = 1000 This definition is not recommended because fine structure of a train can not be seen. This is only used to see long approach phases.
        tdx=0 ## dummy 2008.4.20. For the filtering, tdx is not used. This parameter is for tagging experiment to reduce sounds from other individuals around.
        NPreduction=0
        AllTrain=0
        OKTrain=0
        OKPiDiv=0
        OKtdDiv=0

        print(" ")
        print("-------------------------------------------------------------")
        print("TRAIN BY TRAIN FILTERING")
        print(" ")
        print("1st filter: {} =< Number of clicks in a train (Np) <= ". format(NpTrainMin))
        print("2nd filter:  Variance of Pulse interval (sigma/average of PI) =< {}".format(PiVar))
        print("3rd filter:  Variance of Angle (sigma/average of td)  =< {}".format(tdVar))
        print("4th filter:  Variance of Sound Pressure (sigma/av of SP     =< {} ".format(SPLVar))
        print("5th filter:  Accumulated change of SP in a train   =< {}".format(SPLDif))
        print("6th filter:  Mean IPI should be lower than preset {} value and Median IPI should be lower than preset {} value".format(AvPiMax, MedPiMax))
        print(" ")
        print("Note that large numbers such as 1e*06 means corresponding fintering functoin disabled")

class alts_plot:
    @staticmethod
    def plot_alts_result(dff=None, dfr=None, time_diff=0, atag=2):
        if dff is not None:
            dff = dff[['datetime', 'SPL1', 'time', 'SPL2', 'td', 'PulseInterval', 'SPLR']].copy()
            if atag == 2 and dfr is not None:
                dfr = dfr[['datetime', 'SPL1', 'time', 'SPL2', 'td', 'PulseInterval', 'SPLR']].copy()
            dff['datetime'] = dff['datetime'] + dt.timedelta(seconds=time_diff)
            dff['time'] = dff['time'] + time_diff
            dff_filtered = dff[abs(dff['td']) >= 3]
        if atag == 2:
            dfr_filtered = dfr[abs(dfr['td']) > 3]
        dates = np.array(dff_filtered['datetime'], dtype=np.datetime64)
        data = dict(dff_filtered)
        data = ColumnDataSource(data)
        if atag == 2:
            data2 = dict(dfr_filtered)
            data2 = ColumnDataSource(data2)
        TOOLTIPS = [("time", "@{time}{0.0000f}")]
        TOOLS = "hover,pan, wheel_zoom, box_zoom,reset,save, lasso_select"

        # SPL
        p_spl = figure(width=1000, height=200, x_axis_label='Datetime', y_axis_label='SPL (relative to Pa)',
                       background_fill_color="#fafafa", x_axis_type='datetime', x_range=(dates[0], dates[round(len(dff)/10)]),
                       tools=TOOLS, tooltips=TOOLTIPS)
        p_spl.circle(x='datetime', y='SPL1', size=3, alpha=0.5, fill_color='blue', line_width=0, source=data)

        # SPL Ratio
        p_splr = figure(width=1000, height=150, x_axis_label='Datetime', y_axis_label='SPL Ratio',
                        tools=TOOLS, background_fill_color="#fafafa", x_axis_type='datetime', y_range=(0, 1.5), x_range=p_spl.x_range)
        p_splr.circle(x='datetime', y='SPLR', size=3, alpha=0.5, fill_color='blue', line_width=0, source=data)
        hline2 = Span(location=0.8, dimension='width', line_color='red', line_width=1)
        p_splr.renderers.extend([hline2])

        # Pulse Difference
        TOOLTIPS2 = [("date", "@datetime"), ("time", "@{time}{0.0000f}"), ('td', '@td{0.0f}')]
        p_td = figure(width=1000, height=300, x_axis_label='Datetime', y_axis_label='Time Difference',
                      background_fill_color="#fafafa", x_axis_type='datetime', y_range=(-500, 500), x_range=p_spl.x_range,
                      tools=TOOLS, tooltips=TOOLTIPS2)
        p_td.circle(x='datetime', y='td', size=3, alpha=0.5, fill_color='blue', line_width=0, source=data)

        # Pulse Interval
        p_pulint = figure(width=1000, height=150, x_axis_label='Datetime', y_axis_label='Pulse Interval (ms)',
                          background_fill_color="#fafafa", x_axis_type='datetime', y_axis_type="log", x_range=p_spl.x_range,
                          tools=TOOLS)
        p_pulint.circle(x='datetime', y='PulseInterval', size=3, alpha=0.5, fill_color='blue', line_width=0, source=data)

        if atag == 2:
            p_spl.circle(x='datetime', y='SPL1', size=3, alpha=0.5, fill_color='firebrick', line_width=0, source=data2)
            p_splr.circle(x='datetime', y='SPLR', size=3, alpha=0.5, fill_color='firebrick', line_width=0, source=data2)
            p_td.circle(x='datetime', y='td', size=3, alpha=0.5, fill_color='firebrick', line_width=0, source=data2)
            p_pulint.circle(x='datetime', y='PulseInterval', size=3, alpha=0.5, fill_color='firebrick', line_width=0, source=data2)

        # Selection Box
        select = figure(title="Drag the middle and edges of the selection box to change the range below",
                        height=100, width=1000, y_range=p_spl.y_range,
                        x_axis_type="datetime", y_axis_type=None,
                        tools="", toolbar_location=None, background_fill_color="#efefef")

        range_tool = RangeTool(x_range=p_spl.x_range)
        range_tool.overlay.fill_color = "navy"
        range_tool.overlay.fill_alpha = 0.2

        select.circle('datetime', 'SPL1', source=data)
        select.ygrid.grid_line_color = None
        select.add_tools(range_tool)
        select.toolbar.active_multi = range_tool

        # Statistics display
        stats_div = Div(width=1000)
        stats_div2 = Div(width=1000) if atag == 2 else None

        # CustomJS for updating statistics
        stats_code = """
        const inds = cb_obj.indices;
        const data = source.data;
        let sum_SPL1 = 0, sum_SPL2 = 0, sum_PI = 0, count = inds.length;
        let smallest_time = null; // Initialize to store the smallest time value
        let smallest_datetime = null; // Initialize to store the smallest datetime value

        for (let i = 0; i < count; i++) {
            sum_SPL1 += data['SPL1'][inds[i]];
            sum_SPL2 += data['SPL2'][inds[i]];
            sum_PI += data['PulseInterval'][inds[i]];
            // Update the smallest time
            if (smallest_time === null || data['time'][inds[i]] < smallest_time) {
                smallest_time = data['time'][inds[i]];
            }
            // Update the smallest datetime
            if (smallest_datetime === null || data['datetime'][inds[i]] < smallest_datetime) {
                smallest_datetime = data['datetime'][inds[i]];
            }
        }

        const mean_SPL1 = sum_SPL1 / count;
        const mean_SPL2 = sum_SPL2 / count;
        const SPL_ratio = mean_SPL1 / mean_SPL2;
        const mean_PI = sum_PI / count;

        // Prepare display text for the smallest time and datetime
        let time_text = smallest_time !== null ? `Time: ${smallest_time}<br>` : "";
        let datetime_text = smallest_datetime !== null ? `Datetime: ${new Date(smallest_datetime).toLocaleString('en-US', {timeZone: 'UTC'})}<br>` : "";

        stats_div.text = `<b>Selected Points Statistics:</b><br>
                        ${datetime_text}
                        ${time_text}
                        Mean SPL1: ${mean_SPL1.toFixed(2)}<br>
                        Mean SPL2: ${mean_SPL2.toFixed(2)}<br>
                        SPL Ratio: ${SPL_ratio.toFixed(2)}<br>
                        Mean Pulse Interval: ${mean_PI.toFixed(2)} ms<br>
                        Count: ${count}`;
        """

        data.selected.js_on_change('indices', CustomJS(args={'source': data, 'stats_div': stats_div}, code=stats_code))
        if atag == 2:
            data2.selected.js_on_change('indices', CustomJS(args={'source': data2, 'stats_div': stats_div2}, code=stats_code))

        # Layout
        if atag == 1:
            layout = row(column(p_spl, p_splr, p_td, select, p_pulint), column(stats_div))
        if atag == 2:
            layout = row(column(p_spl, p_splr, p_td, select, p_pulint), column(stats_div, stats_div2))

        show(layout)

    def spl_time_plot(dff, dfr):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        axs[0].plot(dff['datetime'], dff['SPL1'], label='SPL1', color='blue', alpha=0.6)
        axs[0].plot(dff['datetime'], dff['SPL2'], label='SPL2', color='red', alpha=0.6)
        axs[0].set_title('Front A-tag SPL over Time')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('SPL')
        axs[0].legend(frameon=False)
        axs[0].xaxis.set_major_locator(mdates.AutoDateLocator())
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        axs[0].tick_params(axis='x', rotation=45)

        axs[1].plot(dfr['datetime'], dfr['SPL1'], label='SPL1', color='blue', alpha=0.6)
        axs[1].plot(dfr['datetime'], dfr['SPL2'], label='SPL2', color='red', alpha=0.6)
        axs[1].set_title('Rear A-tag SPL over Time')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('SPL')
        axs[1].legend(frameon=False)
        axs[1].xaxis.set_major_locator(mdates.AutoDateLocator())
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        axs[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def spl_time_distribution_plot(df):
        df['hour'] = df['datetime'].dt.hour
        # df.groupby('hour')['SPL1'].mean().plot(kind='bar')
        sns.boxplot(data=df, x='hour', y='SPL1', color='blue', showfliers=False)
        plt.title('Average SPL1 by Hours')
        plt.xlabel('Hour')
        plt.ylabel('Average SPL1')
        plt.show()

    def spl_distribution_plot(df, xlim_SPLR=None, binwidth_SPLR=None):
        df_clean = df[np.isfinite(df['SPLR'])]
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
        
        sns.histplot(x='SPL1', data=df_clean, ax=axs[0], binwidth=50, edgecolor='black')
        axs[0].set_title('SPL1')

        sns.histplot(x='SPL2', data=df_clean, ax=axs[1], binwidth=50, edgecolor='black')
        axs[1].set_title('SPL2')

        binwidth_used = binwidth_SPLR if binwidth_SPLR is not None else 0.1
        axs[2].hist(df_clean['SPLR'], bins=int((df_clean['SPLR'].max() - df_clean['SPLR'].min()) / binwidth_used), edgecolor='black')
        axs[2].set_xlim(xlim_SPLR if xlim_SPLR else (0, 2))
        axs[2].set_title('SPL ratio')


        plt.tight_layout()
        plt.show()

    def spl_cross_corr(df):
        from scipy.signal import correlate
        spl1 = df['SPL1'].dropna()
        spl2 = df['SPL2'].dropna().reindex(spl1.index, method='nearest')

        cross_corr = correlate(spl1 - spl1.mean(), spl2 - spl2.mean(), mode='full')
        lags = np.arange(-len(spl1) + 1, len(spl2))

        plt.figure()
        plt.plot(lags, cross_corr)
        plt.title('Cross-Correlation between SPL1 and SPL2')
        plt.xlabel('Lag')
        plt.ylabel('Cross-correlation')
        plt.show()
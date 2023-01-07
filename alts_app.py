import customtkinter
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
from bokeh.layouts import column, gridplot
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import ColumnDataSource, RangeTool, Span, HoverTool, Slider, DataTable, TableColumn, SelectEditor, NumberFormatter
pd.set_option('display.float_format', '{:.5f}'.format)

customtkinter.set_appearance_mode('dark')
customtkinter.set_default_color_theme('dark-blue')

def file_choose1():
    global file_path1
    file_path1 = filedialog.askopenfilename()

def file_choose2():
    global file_path2
    file_path2 = filedialog.askopenfilename()

def pulse_interval(df):
    df = df.sort_values('datetime', ascending=True)
    df['datetime_next'] = df['datetime'].shift(-1)
    df['PulseInterval'] = (df['datetime_next'] - df['datetime'])/ np.timedelta64(1, 's')
    return df

def load_csv(file_path, position='f'):
    df = pd.read_csv(file_path)
    if len(df.columns)==4:
        df.columns = ['datetime', 'SPL1', 'SPL2', 'td']
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s', origin='1904-01-01')
        df['time'] = df['datetime'].dt.strftime('%H:%M:%S.%f')
        df[['h','m','s']]= [x for x in df['time'].str.split(':')]
        df['time'] = df['h'].astype(int)*3600 + df['m'].astype(int)*60 + df['s'].astype(float)
        df = df[['datetime', 'time', 'SPL1', 'SPL2', 'td']]
    elif len(df.columns)==5:
        df.columns=['date', 'time', 'SPL1', 'SPL2', 'td']
        df['time_f'] = pd.to_datetime(df['time'], unit='s').dt.strftime('%H:%M:%S.%f')
        df['datetime'] = df['date'] + df['time_f'].astype(str)
        df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%Y%H:%M:%S.%f')
    df['SPLR'] = df['SPL1']/df['SPL2']
    if position=='r':
        df = df.rename(columns={"SPL1": "SPL1B", "SPL2": "SPL2B", "td":'tdB'})
    return df

def plot_alts_result(filtered=True, dff=None, dfr=None, time_diff=0, atag=2):
    if filtered == True:
        dff = dff[['datetime', 'SPL1', 'time', 'SPL2', 'td', 'PulseInterval', 'SPLR']]
        if atag==2:
            dfr = dfr[['datetime', 'SPL1', 'time', 'SPL2', 'td', 'PulseInterval', 'SPLR']]
            dfr.columns = ['datetime', 'SPL1B', 'time', 'SPL2B', 'tdB', 'PulseInterval', 'SPLR']
    else:
        dff = dff[['datetime', 'SPL1', 'time', 'SPL2', 'td', 'PulseInterval', 'SPLR']]
        if atag==2:
            dfr = dfr[['datetime', 'SPL1B', 'time', 'SPL2B', 'tdB', 'PulseInterval', 'SPLR']]

    dff['datetime'] = dff['datetime'] + dt.timedelta(seconds=time_diff)
    dff['time'] = dff['time'] + time_diff
    dff_filtered = dff[abs(dff['td'])>=3]
    if atag==2:        
        dfr_filtered = dfr[abs(dfr['tdB'])>=3]
    dates = np.array(dff_filtered['datetime'], dtype=np.datetime64)
    ################################################################
    data = dict(dff_filtered)
    data = ColumnDataSource(data)
    if atag==2:
        data2 = dict(dfr_filtered)
        data2 = ColumnDataSource(data2)
    TOOLTIPS = [("time", "@{time}{0.0000f}")]
    TOOLS = "hover,pan,wheel_zoom,box_zoom,reset,save, lasso_select"
    # SPL
    p_spl = figure(width=1000, height=200, x_axis_label='Datetime', y_axis_label='SPL (relative to Pa)',
            background_fill_color="#fafafa", x_axis_type='datetime', x_range=(dates[0], dates[round(len(dff)/10)]),
            tools=TOOLS, tooltips=TOOLTIPS)
    p_spl.circle(x='datetime',  y='SPL1', size=3, alpha=0.5, fill_color='blue', line_width=0, source=data)
    ################################################################
    # SPL Ratio
    p_splr = figure(width=1000, height=150, x_axis_label='Datetime', y_axis_label='SPL Ratio',
            tools=TOOLS, background_fill_color="#fafafa", x_axis_type='datetime', y_range=(0, 1.5), x_range=p_spl.x_range)
    p_splr.circle(x='datetime',  y='SPLR', size=3, alpha=0.5, fill_color='blue', line_width=0, source=data)
    # Horizontal line
    # hline1 = Span(location=1.5, dimension='width', line_color='red', line_width=1)
    hline2 = Span(location=0.8, dimension='width', line_color='red', line_width=1)
    p_splr.renderers.extend([hline2]) #hline1
    ################################################################
    # Pulse Difference
    TOOLTIPS2 = [("date", "@datetime"), ("time", "@{time}{0.0000f}"), ('td', '@td{0.0f}'), ('tdB', '@tdB{0.0f}')]
    p_td = figure(width=1000, height=300, x_axis_label='Datetime', y_axis_label='Time Difference',
            background_fill_color="#fafafa", x_axis_type='datetime', y_range=(-500, 500), x_range=p_spl.x_range,
            tools=TOOLS, tooltips=TOOLTIPS2)
    p_td.circle(x='datetime', y='td', size=3, alpha=0.5,  fill_color='blue', line_width=0, source=data)        
    ################################################################
    # Pulse Interval
    p_pulint = figure(width=1000, height=150, x_axis_label='Datetime', y_axis_label='Pulse Interval (ms)',
            background_fill_color="#fafafa", x_axis_type='datetime', y_axis_type="log", x_range=p_spl.x_range,
            tools=TOOLS)
    p_pulint.circle(x='datetime', y='PulseInterval', size=3, alpha=0.5, fill_color='blue', line_width=0, source=data)
    if atag==2:
        p_spl.circle(x='datetime',  y='SPL1B', size=3, alpha=0.5, fill_color='firebrick', line_width=0, source=data2)
        p_splr.circle(x='datetime',  y='SPLR', size=3, alpha=0.5, fill_color='firebrick', line_width=0, source=data2)
        p_td.circle(x='datetime', y='tdB', size=3, alpha=0.5,  fill_color='firebrick', line_width=0, source=data2)
        p_pulint.circle(x='datetime', y='PulseInterval', size=3, alpha=0.5, fill_color='firebrick', line_width=0, source=data2)
    ################################################################
    # Selection Box
    select = figure(title="Drag the middle and edges of the selection box to change the range below",
                    height=100, width=1000, y_range=p_spl.y_range,
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="#efefef")

    range_tool = RangeTool(x_range=p_spl.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    select.circle('datetime', 'SPL1', source=dff_filtered)
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool
    ################################################################
    # time_change = Slider(start=0.1, end=0.2, value=0.15, step=0.001, title='Time Change')
    ################################################################
    show(column(p_spl, p_splr, p_td, select,  p_pulint))

def run_analysis():
    dff = load_csv(file_path1, position='f')
    dfr = load_csv(file_path2, position='r')
    dff.SPL1 = dff.SPL1*0.771
    dff.SPL2 = dff.SPL2*0.771
    print("###  FRONT data loading completed ###")
    dff = pulse_interval(dff)

    dfr.SPL1B = dfr.SPL1B*0.771
    dfr.SPL2B = dfr.SPL2B*0.771
    print("###  REAR data loading completed ###")
    dfr = pulse_interval(dfr)
    print('### Pulse Interval Completed ###')
    
    plot_alts_result(filtered=False, dff=dff, dfr=dfr, time_diff=0, atag=2)


root = customtkinter.CTk()
root.title('Acoustic Line Transect Result Plot - by Y-W HO')
root.geometry("400 x 500")

frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill='both', expand=True)

label1 = customtkinter.CTkLabel(master=frame, text='Front A-tag csv import')
label1.pack(pady=12, padx=10)

button1 = customtkinter.CTkButton(master=frame, text='Choose CSV', command=file_choose1)
button1.pack(pady=12, padx=10)

label2 = customtkinter.CTkLabel(master=frame, text='Back A-tag csv import')
label2.pack(pady=12, padx=10)

button2 = customtkinter.CTkButton(master=frame, text='Choose CSV', command=file_choose2)
button2.pack(pady=12, padx=10)

entry1 = customtkinter.CTkEntry(master=frame, placeholder_text='Front A-tag time')
entry1.pack(pady=12, padx=10)

entry2 = customtkinter.CTkEntry(master=frame, placeholder_text='Back A-tag time')
entry2.pack(pady=12, padx=10)

button3 = customtkinter.CTkButton(master=frame, text='Plot Results', command=run_analysis)
button3.pack(pady=12, padx=10)

root.mainloop()
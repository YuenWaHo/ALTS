import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
from bokeh.layouts import column, gridplot
from bokeh.plotting import figure, show, output_file, save, reset_output, output_notebook
from bokeh.models import ColumnDataSource, RangeTool, Span, HoverTool, Slider, DataTable, TableColumn, SelectEditor, NumberFormatter
pd.set_option('display.float_format', '{:.5f}'.format)

class alts_load:
    def alts_analysis(atag=2):
        dff = alts_load.load_csv(df1, position='f')
        dff.SPL1 = dff.SPL1*0.771
        dff.SPL2 = dff.SPL2*0.771
        st.write("###  FRONT data loading completed ###")
        dff = alts_load.pulse_interval(dff)

        if atag==2:
            dfr = alts_load.load_csv(df2, position='r')
            dfr.SPL1B = dfr.SPL1B*0.771
            dfr.SPL2B = dfr.SPL2B*0.771
            st.write("###  REAR data loading completed ###")
            dfr = alts_load.pulse_interval(dfr)
            st.write('### Pulse Interval Completed ###')
            return dff, dfr
        else:
            st.write('### Pulse Interval Completed ###')
            return dff
        
    def load_csv(df, position='f'):
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

    def pulse_interval(df):
        df = df.sort_values('datetime', ascending=True)
        df['datetime_next'] = df['datetime'].shift(-1)
        df['PulseInterval'] = (df['datetime_next'] - df['datetime'])/ np.timedelta64(1, 's')
        return df

class alts_filter:
    def filter_condition(noise_condition='1'):
        SPLthreshold = 43
        if noise_condition==1:
            print("Raw viewer without any filter")
        elif noise_condition==2:
            print("Recommended for most of the cases with a mild filter")
            ### PULSE BY PULSE FILTERING  parameter setting ##
            ## Exclude td=0 and small SPL lower than SPLthreshold (SPL=SPLthreshold is accepted)
            tdswitch=1                 ##  1;exclude single trigger, 0;include single trigger
            Reflection=0.5             ## exclude reflections < Reflection (ms) from the previous pulse
            Reflection=Reflection/1000 ## convert to second unit
            SPLR=0                     ## species descrimination. 0=Disabled. Extract SPL1/SPL2>=SPLR to focus on high frequency clicks. DISABLED;0

            ## Exclude noise from snapping shrimps or other isolated pulse sounds.
            Isolation=1000000          ## exclude independent pulses isolated >  Isolation (ms) from the both sides' pulses
                                    ## include independent pulses isolated <= Isolation (ms) from the both sides' pulses
            Isolation=Isolation/1000   ## convert to second unit

            ## Smoothing both for pulse interval and sound pressure         
            smoothwin =3               ## extract 1/smoothwin <= Pi difference <= smoothwin
            smoothSPL =1000000         ## Disabled. SPL smoothing makes noise as signal

            ## TRAIN BY TRAIN FILTERING  parameter setting-------------------------------------------------------------
            ## screening in a train
            NpTrainMin=5               ##  >= minimum number of pulses in a train
            NpTrainMax=1000000         ## Disabled, <= maximum number of pulses in a train to exclude ship noise
            PiVar     =0.4             ##  <= SdPi/AvPi should be smaller than this
            tdVar     =1000000         ## Disabled. <= Sdtd/511(td max.) should be smaller than this (localized within small bearing angle)
            SPLDif    =1000000         ## Disabled. <= (accumulated difference of SPL/number of pulse in a train) / AvPow 
            SPLVar    =1000000         ## Disabled. <= SdSPL/AvSPL should be smaller than this
            AvPiMax   =1000000         ## Disabled. <= AvPi should be smaller than this
            MedPiMax  =1000000         ## Disabled. <= MedPiMax should be smaller than this
        elif noise_condition==3:
            print('For very noisy condition such as fixed monitoring close to shore with a lot of snapping shrimps')
            ## PULSE BY PULSE FILTERING  parameter setting-------------------------------------------------------------

            ## Exclude td=0 and small SPL lower than SPLthreshold (SPL=SPLthreshold is accepted)
            tdswitch=1                 ##   1;exclude single trigger, 0;include single trigger
            Reflection=2.5             ## exclude reflections < Reflection (ms) from the previous pulse
            Reflection=Reflection/1000 ## convert to second unit
            SPLR=0                     ## species descrimination. 0=Disabled. Extract SPL1/SPL2>=SPLR to focus on high frequency clicks. DISABLED;0

            ## Exclude noise from snapping shrimps or other isolated pulse sounds.
            Isolation=200              ## exclude independent pulses isolated >  Isolation (ms) from the both sides' pulses
                                    ## include independent pulses isolated <= Isolation (ms) from the both sides' pulses
            Isolation=Isolation/1000   ## convert to second unit

            ## Smoothing both for pulse interval and sound pressure         
            smoothwin =3               ## extract 1/smoothwin <= Pi difference <= smoothwin
            smoothSPL =1000000         ## Disabled. SPL smoothing makes noise as signal

            ## TRAIN BY TRAIN FILTERING  parameter setting-------------------------------------------------------------
            ## screening in a train
            NpTrainMin=5               ##  >= minimum number of pulses in a train
            NpTrainMax=1000000         ## Disabled, <= maximum number of pulses in a train to exclude ship noise
            PiVar     =0.5             ##  <= SdPi/AvPi should be smaller than this
            tdVar     =0.5             ## Disabled. <= Sdtd/511(td max.) should be smaller than this (localized within small bearing angle)
            SPLDif    =1000000         ## Disabled. <= (accumulated difference of SPL/number of pulse in a train) / AvPow 
            SPLVar    =1000000         ## Disabled. <= SdSPL/AvSPL should be smaller than this
            AvPiMax   =1000000         ## Disabled. <= AvPi should be smaller than this
            MedPiMax  =1000000         ## Disabled. <= MedPiMax should be smaller than this
        elif noise_condition==4:
            ## Exclude td=0 and small SPL lower than SPLthreshold (SPL=SPLthreshold is accepted)
            tdswitch=1                ##   1;exclude single trigger, 0;include single trigger
            Reflection=2.5             ## exclude reflections < Reflection (ms) from the previous pulse
            SPLR=0                     ## species descrimination. 0=Disabled. Extract SPL1/SPL2>=SPLR to focus on high frequency clicks. DISABLED;0

            ## Exclude noise from snapping shrimps or other isolated pulse sounds.
            Isolation=1000000          ## exclude independent pulses isolated >  Isolation (ms) from the both sides' pulses
                                ## include independent pulses isolated <= Isolation (ms) from the both sides' pulses

            ## Smoothing both for pulse interval and sound pressure         
            smoothwin =1000000         ## extract 1/smoothwin <= Pi difference <= smoothwin
            smoothSPL =1000000         ## Disabled. SPL smoothing makes noise as signal

            ## TRAIN BY TRAIN FILTERING  parameter setting-------------------------------------------------------------
            ## screening in a train
            NpTrainMin=6               ##  >= minimum number of pulses in a train
            NpTrainMax=1000000         ## Disabled, <= maximum number of pulses in a train to exclude ship noise
            PiVar     =1000000         ##  <= SdPi/AvPi should be smaller than this
            tdVar     =1000000         ## Disabled. <= Sdtd/511(td max.) should be smaller than this (localized within small bearing angle)
            SPLDif    =1000000         ## Disabled. <= (accumulated difference of SPL/number of pulse in a train) / AvPow 
            SPLVar    =1000000         ## Disabled. <= SdSPL/AvSPL should be smaller than this
            AvPiMax   =1000000         ## Disabled. <= AvPi should be smaller than this
            MedPiMax  =1000000         ## Disabled. <= MedPiMax should be smaller than this
        else:
            print('Unexpected Noise Condition')
            print("1: No filter applies. Just raw view") 
            print("2: Recommended for most of the cases with a mild filter") 
            print("3: For very noisy condition such as fixed monitoring close to shore with a lot of snapping shrimps") 
            print("4: Advanced parameter setting") 
            print(" ")
        return SPLthreshold, tdswitch, Reflection, Isolation, smoothwin, smoothSPL
    
    def mean_angle(df): #Count number of pulses in time bin
        total_pulse = len(df)
        timebin = 1 #seconds
        n_box = (df.iloc[-1].datetime - df.iloc[0].datetime)/timebin #number of boxes

    def clean_step0(df):
        ## 0th screening; delete td=0
        print('###--- 0th screening ---###')
        original_number_click = len(df)
        if tdswitch==1:
            print("Exclude clicks without source direction information (td=+511 excluded)")
            df = df[(abs(df['td']) > 0) & (df['td']<511)]
            p = len(df)
        elif tdswitch==0:
            print('Include single trigger data without bearing angle information (include td=0)')
            df = df[(df['td']<511)]
            p = len(df)
        print("Total number of pulses: {}  ; survived after excluding td=0: {}".format(original_number_click, p))
        print("")
        return df
        
    def cleanup(df, pos='front'):
        print("----------PULSE BY PULSE FILTERING----------")
        if pos=='rear':
            df = df[['datetime', 'SPL1B', 'SPL2B', 'tdB', 'PulseInterval', 'time', 'SPLR']]
            df.columns = ['datetime', 'SPL1', 'SPL2', 'td', 'PulseInterval', 'time', 'SPLR']
        else:
            df = df[['datetime', 'SPL1', 'SPL2', 'td', 'PulseInterval', 'time', 'SPLR']]

        ## 0th screening; delete td=0
        print('###--- 0th screening ---###')
        original_number_click = len(df)
        if tdswitch==1:
            print("Exclude clicks without source direction information (td=+511 excluded)")
            df = df[(abs(df['td']) > 0) & (df['td']<511)]
            p = len(df)
        elif tdswitch==0:
            print('Include single trigger data without bearing angle information (include td=0)')
            df = df[(df['td']<511)]
            p = len(df)
        print("Total number of pulses: {}  ; survived after excluding td=0: {}".format(original_number_click, p))
        print("")

        ##1st screening; eliminate low intensity and reflections
        print('###--- 1st screening ---###')
        print ('Select clicks with sound pressure >= {} counts. 1count=0.077Pa approximately'.format(SPLthreshold))
        print ("Eeliminated possible reflections associated within {} ms after the previous pulse".format(Reflection*1000))
        total_pulse = len(df)
        df = df[(df['SPL1'] >= SPLthreshold) & (df['PulseInterval'] > Reflection)]
        p = len(df)
        print("Total number of pulses: {}  ; survived after threshold and reflection filter: {}".format(total_pulse, p))
        print("")

        ##2nd screening; eliminate isolated pulse
        print('###--- 2nd screening ---###')
        print('Eliminated isolated pulses {} ms apart from both sides pulses'.format(Isolation*1000))
        total_pulse = len(df)
        df['datetime_front'] = df['datetime'].shift(1)
        df['Pulse_time_diff_front'] = abs(df['datetime_front'] - df['datetime'])/ np.timedelta64(1, 's')
        df['datetime_back'] = df['datetime'].shift(-1)
        df['Pulse_time_diff_back'] = abs(df['datetime_back'] - df['datetime'])/ np.timedelta64(1, 's')
        df = df[(df['Pulse_time_diff_back'] <= Isolation) & (df['Pulse_time_diff_front'] <= Isolation)]
        p = len(df)
        print("Total number of pulses: {}  ; survived after isolated pulse filter: {}".format(total_pulse, p))
        print("")
        df = df[['datetime', 'SPL1', 'time','SPL2', 'td', 'PulseInterval', 'SPLR']]

        ##3rd screening; eliminate unusual change of inter-pulse intervals
        print('###--- 3rd screening ---###')
        print ("Select clicks with smoothly changed inter-pulse intervals with R=< {}".format(smoothwin))
        print ("where R=(IPI present)/(IPI previous).   Accept  1/smoothIPI < R < smoothIPI") 
        total_pulse = len(df)
        df['PI_next'] = df['PulseInterval'].shift(-1)
        df['R'] = df['PulseInterval']/df['PI_next']
        df = df[(df['R']<=smoothwin) & (df['R']>1/smoothwin)]
        p = len(df)  
        print("Total number of pulses: {}  ; survived after smoothing IPI: {}".format(total_pulse, p))
        print("")
        df = df[['datetime', 'SPL1', 'time','SPL2', 'td', 'PulseInterval', 'SPLR']]

        ##4th screening; eliminate unusual change of SPL
        print('###--- 4th screening ---###')
        print ("Select clicks with smoothly changed sound pressure with R=< {}".format(smoothSPL))
        print ("where R=(SPL present)/(SPL previous).  Accept  1/smoothSPL < R < smoothSPL") 
        total_pulse = len(df)
        df['SPL1_next'] = df['SPL1'].shift(-1)
        df['R'] = df['SPL1']/df['SPL1_next']
        df = df[(df['R']<=smoothSPL) & (df['R']>1/smoothSPL)]
        p = len(df)  
        print("Total number of pulses: {}  ; survived after smoothing SPL: {}".format(total_pulse, p))
        print("")
        df = df[['datetime', 'SPL1', 'time','SPL2', 'td', 'PulseInterval', 'SPLR']]
        
        print(" ")
        print("Total number of pulses in the original data file = {} ".format(original_number_click))
        print('{} pulses survived after clicl-by-click filtering'.format(p))
        print("Note that large numbers such as 1e*06 means the filtering functoin disabled")
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
            dfr_filtered = dfr[abs(dfr['tdB'])>3]
        dates = np.array(dff_filtered['datetime'], dtype=np.datetime64)

        data = dff_filtered.copy()
        data2 = dfr_filtered.copy()

        # data = dict(dff_filtered)
        # data = ColumnDataSource(data)
        # if atag==2:
        #     data2 = dict(dfr_filtered)
        #     data2 = ColumnDataSource(data2)
        TOOLTIPS = [("time", "@{time}{0.0000f}")]
        TOOLS = "hover,pan,wheel_zoom,box_zoom,reset,save, lasso_select"
        # SPL
        p_spl = figure(width=1000, height=200, x_axis_label='Datetime', y_axis_label='SPL (relative to Pa)',
                background_fill_color="#fafafa", x_axis_type='datetime', x_range=(dates[0], dates[round(len(dff)/10)]),
                tools=TOOLS, tooltips=TOOLTIPS)
        p_spl.circle(x='datetime',  y='SPL1', size=3, alpha=0.5, fill_color='blue', line_width=0, source=data)
        ################################################################
        splr_df = dict(dff_filtered)
        splr_data = ColumnDataSource(splr_df)
        # SPL Ratio
        p_splr = figure(width=1000, height=150, x_axis_label='Datetime', y_axis_label='SPL Ratio',
                tools=TOOLS, background_fill_color="#fafafa", x_axis_type='datetime', y_range=(0, 1.5), x_range=p_spl.x_range)
        p_splr.circle(x='datetime',  y='SPLR', size=3, alpha=0.5, fill_color='blue', line_width=0, source=splr_data)
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
        # show(column(p_spl, p_splr, p_td, select,  p_pulint))
        plot1 = st.bokeh_chart(p_spl, use_container_width=True)
        plot2 = st.bokeh_chart(p_splr, use_container_width=True)
        # st.bokeh_chart(p_td)
        # st.bokeh_chart(select)
        # st.bokeh_chart(p_pulint)
        # return p_spl, p_splr, p_td, p_pulint, select
        return plot1, plot2



st. set_page_config(layout="wide")
st.title('Acoustic Line Transect Analysis Toolkit')
st.markdown("The site can be used to visualize acoustic line-transect data." )

uploaded_file_front_atag = st.file_uploader("Front A-tag csv file")
if uploaded_file_front_atag is not None:
    df1 = pd.read_csv(uploaded_file_front_atag, header=None)
    # if st.checkbox('Show raw data', key='CB1'):
    #     st.subheader('Raw data')
    #     st.write(df1)

uploaded_file_rear_atag = st.file_uploader("Rear A-tag csv file")
if uploaded_file_rear_atag is not None:
    df2 = pd.read_csv(uploaded_file_rear_atag, header=None)
    # if st.checkbox('Show raw data', key='CB2'):
    #     st.subheader('Raw data')
    #     st.write(df2)

if st.checkbox('Process A-tag data'):
    dff, dfr = alts_load.alts_analysis(atag=2)
else:
    st.write('Please upload file again')


st.write(dff)
st.write(dfr)
st.write('')
st.write('')
st.write('')

col1, col2 = st.columns(2)
with col1:
    filtered_button = st.checkbox('Filtered')
with col2:
    number_atag = st.checkbox('2 A-tags')

col1, col2 = st.columns(2)
with col1:
    front_atag_time = st.number_input('Insert front A-tag time')
with col2:
    back_atag_time = st.number_input('Insert back A-tag time')
    time_diff = back_atag_time - front_atag_time
    st.write(time_diff)

if filtered_button:
    dff = dff[['datetime', 'SPL1', 'time', 'SPL2', 'td', 'PulseInterval', 'SPLR']]
    if number_atag:
        dfr = dfr[['datetime', 'SPL1', 'time', 'SPL2', 'td', 'PulseInterval', 'SPLR']]
        dfr.columns = ['datetime', 'SPL1B', 'time', 'SPL2B', 'tdB', 'PulseInterval', 'SPLR']
else:
    dff = dff[['datetime', 'SPL1', 'time', 'SPL2', 'td', 'PulseInterval', 'SPLR']]
    if number_atag:
        dfr = dfr[['datetime', 'SPL1B', 'time', 'SPL2B', 'tdB', 'PulseInterval', 'SPLR']]

dff['datetime'] = dff['datetime'] + dt.timedelta(seconds=time_diff)
dff['time'] = dff['time'] + time_diff
dff_filtered = dff[abs(dff['td'])>=3]
if number_atag:        
    dfr_filtered = dfr[abs(dfr['tdB'])>3]
dates = np.array(dff_filtered['datetime'], dtype=np.datetime64)

TOOLTIPS = [("time", "@{time}{0.0000f}")]
TOOLS = "hover,pan,wheel_zoom,box_zoom,reset,save, lasso_select"
# SPL
p_spl = figure(width=1000, height=200, x_axis_label='Datetime', y_axis_label='SPL (relative to Pa)',
        background_fill_color="#fafafa", x_axis_type='datetime', x_range=(dates[0], dates[round(len(dff)/10)]),
        tools=TOOLS, tooltips=TOOLTIPS)
p_spl.circle(x='datetime',  y='SPL1', size=3, alpha=0.5, fill_color='blue', line_width=0, source=dff)
p_spl.circle(x='datetime',  y='SPL1', size=3, alpha=0.5, fill_color='firebrick', line_width=0, source=dfr)
st.bokeh_chart(p_spl)

p_splr = figure(width=1000, height=150, x_axis_label='Datetime', y_axis_label='SPL Ratio',
            tools=TOOLS, background_fill_color="#fafafa", x_axis_type='datetime', y_range=(0, 1.5), x_range=p_spl.x_range)
p_splr.circle(x='datetime',  y='SPLR', size=3, alpha=0.5, fill_color='blue', line_width=0, source=dff)
p_splr.circle(x='datetime',  y='SPLR', size=3, alpha=0.5, fill_color='firebrick', line_width=0, source=dfr)
# Horizontal line
# hline1 = Span(location=1.5, dimension='width', line_color='red', line_width=1)
hline2 = Span(location=0.8, dimension='width', line_color='red', line_width=1)
p_splr.renderers.extend([hline2]) #hline1
st.bokeh_chart(p_splr)
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta 
import os
import shutil
import config

KPA_TO_CM = 10.197
PA_TO_CM = 0.010197
FT_TO_CM = 30.48

def infer_site_from_name(fn):
    fn_lc = fn.lower()
    sitekey = None
    
    if 'lyell' in fn_lc and 'maclure' in fn_lc:
        sitekey = 0
    elif 'lyell' in fn_lc and ('twin' in fn_lc or 'tb' in fn_lc):
        sitekey = 1
    elif 'dana' in fn_lc:
        sitekey = 2
    elif 'tuolumne' in fn_lc or 'tum' in fn_lc or 'tr' in fn_lc:
        sitekey = 3
    elif 'budd' in fn_lc:
        sitekey = 4
    elif 'delaney' in fn_lc:
        sitekey = 5
    else:
        print('Unable to infer site from given fn')
        
    return sitekey


def infer_sensor_type_from_name(fn):
    fn_lc = fn.lower()
    
    # assume sensor is levelogger (sensor type defaults to levelogger
    # since some older levelogger files do not include level in filename
    # and barologger are expected to include baro in the name)
    return ('lvl' if ('baro' not in fn_lc) else 'baro')

def read_meta(fn):
    last_read = -1
    meta_data = []
    
    f = open(fn, 'r', encoding='latin1')
    
    found_header = False
    while(not found_header):
        line = f.readline()
        if line != '\n' and line.strip() != '':
            last_read += 1
            meta_data.append(line)
        
        found_header = 'date' in line.lower()  
        
    f.close()
    header_line_num = last_read
    
    return meta_data, header_line_num

def read_normalize_solinst_data(fn, header_num, sensor_type):
    df = pd.read_csv(fn,
                     header=header_num,
                     usecols=[0, 1, 3, 4],
                     index_col=0,
                     parse_dates=[[0,1]],
                     infer_datetime_format=True,
                     encoding_errors='ignore')
        
    if sensor_type == 'lvl':
        df.columns = config.LEVEL_HEADER
    else:
        df['BAROMETRIC PRESSURE'] *= KPA_TO_CM
        df.columns = config.BARO_HEADER
    
    return df

def load_normalized_solinst_data(fn):
    return pd.read_csv(fn, index_col=0, parse_dates=[0], infer_datetime_format=True)

def plot_solinst_pressure_temp(df, site_info, ax=None, sensor_type='lvl', figsize=config.FIGSIZE):
    sensor_label = 'Levelogger' if (sensor_type == 'lvl') else 'Barologger'
    
    if ax is None:
        fig, ax = plt.subplots(1, 2, sharex=True, figsize=figsize)
    
    plt.suptitle(f'{site_info["site"]} {site_info["span"]}')
    
    ax[0].plot(df[sensor_type + '_cm'])
    ax[0].set_ylabel('cm')
    ax[0].set_xlabel('Date')
    ax[0].set_title(f'Solinst {sensor_label} Raw Pressure')
    
    ax[1].plot(df['temp_C'])
    ax[1].set_ylabel('Degrees C')
    ax[1].set_xlabel('Date')
    ax[1].set_title(f'Solinst {sensor_label} Raw Temperature')
    
def plot_temp_interval(df, start, end, site_info, label=None, figsize=config.FIGSIZE):
    if label is None:
        title = f'{site_info["site"]} {site_info["span"]} Temperature Interval'
    else:
        title = f'{site_info["site"]} {site_info["span"]} {label}'
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df['temp_C'][start:end])
    ax.set_ylabel('Degrees C')
    ax.set_xlabel('Date Time')
    ax.set_title(title)
    
def plot_pressure_interval(df, start, end, site_info,  sensor_type='lvl', label=None, figsize=config.FIGSIZE):
    sensor_label = 'Levelogger' if (sensor_type == 'lvl') else 'Barologger'
    
    if label is None:
        title = f'{site_info["site"]} {site_info["span"]} {sensor_label} Pressure Interval'
    else:
        title = f'{site_info["site"]} {site_info["span"]} {label}'
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df[sensor_type + '_cm'][start:end])
    ax.set_ylabel('cm')
    ax.set_xlabel('Date Time')
    ax.set_title(title)
    
    
def plot_in_water(df, insert_datetime_str, remove_datetime_str, site_info, sensor_type='lvl', figsize=config.FIGSIZE):
    sensor_label = 'Levelogger' if (sensor_type == 'lvl') else 'Barologger'
    
    # ** INSERT **
    
    insert_datetime = datetime.strptime(insert_datetime_str, '%Y-%m-%d %H:%M')
    insert_start = max(df.index[0], insert_datetime - timedelta(hours=12))
    insert_stop = insert_datetime + timedelta(hours=12)
    
    # plot small interval near beginning of series
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    plt.suptitle(f'{site_info["site"]} {site_info["span"]} Time Series Around Insertion')
    
    ax[0].plot(df[sensor_type + '_cm'][insert_start:insert_stop], label='Full Series')
    ax[0].plot(df[sensor_type + '_cm'][insert_datetime:insert_stop], label='In Water')
    ax[0].set_ylabel('cm')
    ax[0].set_title(f'Solinst {sensor_label} Raw Pressure')
    ax[0].tick_params('x', labelrotation=45)
    ax[0].legend()
    
    
    ax[1].plot(df['temp_C'][insert_start:insert_stop], label='Full Series')
    ax[1].plot(df['temp_C'][insert_datetime:insert_stop], label='In Water')
    ax[1].set_ylabel('Degrees C')
    ax[1].set_title(f'Solinst {sensor_label} Raw Temperature')
    ax[1].legend()
    ax[1].tick_params('x', labelrotation=45)
    
     # ** Remove **
    
    remove_datetime = datetime.strptime(remove_datetime_str, '%Y-%m-%d %H:%M')
    remove_stop = min(df.index[-1], remove_datetime + timedelta(hours=12))
    remove_start = remove_datetime - timedelta(hours=12)
    
    # plot small interval near beginning of series
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    plt.suptitle(f'{site_info["site"]} {site_info["span"]} Time Series Around Removal')
    
    ax[0].plot(df[sensor_type + '_cm'][remove_start:remove_stop], label='Full Series')
    ax[0].plot(df[sensor_type + '_cm'][remove_start:remove_datetime], label='In Water')
    ax[0].set_ylabel('cm')
    ax[0].set_title(f'Solinst {sensor_label} Raw Pressure')
    ax[0].tick_params('x', labelrotation=45)
    ax[0].legend()
    
    ax[1].plot(df['temp_C'][remove_start:remove_stop], label='Full Series')
    ax[1].plot(df['temp_C'][remove_start:remove_datetime], label='In Water')
    ax[1].set_ylabel('Degrees C')
    ax[1].set_title(f'Solinst {sensor_label} Raw Temperature')
    ax[1].legend()
    ax[1].tick_params('x', labelrotation=45)
    
def trim_df(df, insert_datetime_str, remove_datetime_str, site_info, plot=True, figsize=config.FIGSIZE):
    insert_datetime = datetime.strptime(insert_datetime_str, '%Y-%m-%d %H:%M')
    remove_datetime = datetime.strptime(remove_datetime_str, '%Y-%m-%d %H:%M')
    
    mask = (df.index >= insert_datetime) & (df.index <= remove_datetime)
    df_trim = df.loc[mask]
    
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        plot_solinst_pressure_temp(df, site_info, ax=ax)
        plot_solinst_pressure_temp(df_trim, site_info, ax=ax)
        ax[0].legend(['Full Series', 'In Water'])
        ax[1].legend(['Full Series', 'In Water'])
    
    return df_trim

def load_ncar_baro(fn):
    ds = xr.open_dataset(fn) # in pascals
    df = ds.to_dataframe()
    df.drop(columns=['lat', 'lon'], inplace=True)
    df.columns = config.NCAR_BARO_HEADER
    df.index.names = ['Date_Time']
    return df

def plot_ncar_baro(df, site_info, figsize=config.FIGSIZE):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df['baro_cm'])
    ax.set_ylabel('cm')
    ax.set_title(f'{site_info["site"]} {site_info["span"]} NCAR Barometric Pressure')
    
def interpolate_to_timeseries(target_dt_index, input_df):
    column_labels = input_df.columns
    
    frame = {'Date_Time': target_dt_index}
    for column in column_labels:
        interp_s = pd.Series(np.interp(target_dt_index.asi8, input_df.index.asi8, input_df[column]))
        frame[column] = interp_s
        
    output_df = pd.DataFrame(frame)
    output_df = output_df.set_index('Date_Time')
    
    return output_df

def get_next_smallest_30min(t):
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour) + timedelta(minutes=t.minute//30 * 30))

def plot_pres_timeseries(ds, ax=None, figsize=config.FIGSIZE):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title('Pressure Time Series')
    
    ax.plot(ds)
    ax.set_ylabel('cm')
    
    
def plot_multiple_pres_timeseries(series, labels, site_info, figsize=config.FIGSIZE):
    
    fig, ax = plt.subplots(figsize=figsize)
    for s in series:
        plot_pres_timeseries(s, ax=ax)
    ax.legend(labels)
    ax.set_title(f'{site_info["site"]} {site_info["span"]} Pressure Time Series')
    
def get_next_largest_30min(t):
    m = 0
    if t.minute != 0:
        m = 30 if (t.minute <= 30) else 60
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour) + timedelta(minutes=m))

def get_30min_spanning_index(df_to_span):
    start_dt = get_next_smallest_30min(df_to_span.index[0])
    end_dt = get_next_largest_30min(df_to_span.index[-1])
    return pd.date_range(start_dt, end_dt, freq='30min')

def barocorrect_level(level_df, baro_df):
    return level_df['lvl_cm'] - baro_df['baro_cm']

def plot_pres_multiple_interval(ds, label, site_info, figsize=config.FIGSIZE):
    month_start_dt = site_info['collection_year'] + '-6'
    month_end_dt = site_info['collection_year'] + '-8'
    
    week_start_dt = site_info['collection_year'] + '-7'
    week_end_dt = site_info['collection_year'] + '-7-7'
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)

    ax[0].plot(ds)
    ax[0].set_ylabel('cm')
    ax[0].set_title('Whole Series')
    ax[0].tick_params('x', labelrotation=45)

    ax[1].plot(ds[month_start_dt:month_end_dt])
    ax[1].set_ylabel('cm')
    ax[1].set_title('Several Month Duration')
    ax[1].tick_params('x', labelrotation=45)
    
    ax[2].plot(ds[week_start_dt:week_end_dt])
    ax[2].set_ylabel('cm')
    ax[2].set_title('Week Duration')
    ax[2].tick_params('x', labelrotation=45)

    plt.suptitle(f'{site_info["site"]} {site_info["span"]} {label} at Various Time Scales')
    
def plot_candidate(ds, candidate_bounds, site_info, figsize=config.FIGSIZE):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    plt.suptitle(f'{site_info["site"]} {site_info["span"]}')
    
    ax[0].plot(ds)
    ax[0].plot(ds[candidate_bounds['start']: candidate_bounds['end']], label='Ice Jam/Displacement Candidate')
    ax[0].legend()
    ax[0].set_ylabel('cm')
    ax[0].set_title('Ice Jam/Displacement Candidate Relative to Whole Series')
    ax[0].tick_params('x', labelrotation=45)
    
    ax[1].plot(ds[candidate_bounds['start']: candidate_bounds['end']])
    ax[1].set_ylabel('cm')
    ax[1].set_title('Profile of Ice Jam/Displacement Candidate')
    ax[1].tick_params('x', labelrotation=45)
    
def make_output_df(datetime_index, raw_pressure, barocorrected_pressure, temp, ice_jams, sensor_malfunctions=None):
    data = [datetime_index, raw_pressure, barocorrected_pressure, temp, np.zeros(len(raw_pressure), dtype=np.int8)]
    labels = config.BARO_CORRECTED_HEADER
    frame = {l:d for l,d in zip(labels, data)}
    output_df = pd.DataFrame(frame)
    output_df = output_df.set_index(labels[0])
    
    if ice_jams is not None:
        for ice_jam in ice_jams:
            output_df.loc[output_df[ice_jam['start']:ice_jam['end']].index, labels[-1]] = config.ICEJAM_FLAG
    
    if sensor_malfunctions is not None:
        for sm in sensor_malfunctions:
            output_df.loc[output_df[sm['start']:sm['end']].index, labels[-1]] = config.MALFUNCTION_FLAG
    
    return output_df

def split_series_at_shifts(output_df, sensor_shifts):
    # sensor_shifts must be sorted/in order
    
    output_dfs = []
    if sensor_shifts is None:
        output_dfs.append(output_df)
    else:
        last = output_df
        for shift in sensor_shifts:
            output_dfs.append(output_df[:shift['start']])
            last = output_df[shift['end']:]
        output_dfs.append(last)
    
    return output_dfs

def save_barocorrected_timeseries(output_dfs, site_info):
    for i, df in enumerate(output_dfs):
        output_fn = config.BAROCORRECTED_OUTPUT_FN.format(site=config.SITE_SHORTNAME[site_info['sitecode']], 
                                                            year=site_info['collection_year'],
                                                            segment=i)
        output_dir = os.path.join('..', 'data', 'barocorrected')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        output_path = os.path.join(output_dir, output_fn)
        df.to_csv(output_path)
        print(f'Wrote segment {i} to {output_path}')
        
def get_summary(level_path, 
                utc_note, 
                baro_path, 
                ice_jams, 
                sensor_shifts):
    l = f'Level Source: {level_path}\n'
    utc = f'UTC Inspection: {utc_note}\n'
    b = f'Baro Source: {baro_path}\n'
    
    ice = 'Ice Jams:'
    if ice_jams is None:
        ice += ' None\n'
    else:
        for i, ij in enumerate(ice_jams, 1):
            ice += f'\nIce Jam {i}\nstart: {ij["start"]}\nend: {ij["end"]}\n\n'
    
    shift = 'Sensor Shifts:'
    if sensor_shifts is None:
        shift += ' None\n'
    else:
        for i,s in enumerate(sensor_shifts, 1):
            shift += f'\nShift {i}\nstart: {s["start"]}\nend: {s["end"]}\n\n'
    return l + utc + b + ice + shift

def resample_vented(vented_df):
    return vented_df[vented_df.index.minute % 30 == 0]

def make_vented_output_df(datetime_index, lvl, temp, ice_jams, sensor_malfunctions=None):
    data = [datetime_index, lvl, temp, np.zeros(len(lvl), dtype=np.int8)]
    labels = config.VENTED_CORRECTED_HEADER
    frame = {l:d for l,d in zip(labels, data)}
    output_df = pd.DataFrame(frame)
    output_df = output_df.set_index(labels[0])
    
    if ice_jams is not None:
        for ice_jam in ice_jams:
            output_df.loc[output_df[ice_jam['start']:ice_jam['end']].index, labels[-1]] = config.ICEJAM_FLAG
    
    if sensor_malfunctions is not None:
        for sm in sensor_malfunctions:
            output_df.loc[output_df[sm['start']:sm['end']].index, labels[-1]] = config.MALFUNCTION_FLAG
    
    return output_df

def save_vented_timeseries(output_dfs, site_info):
    for i, df in enumerate(output_dfs):
        output_fn = config.VENTED_OUTPUT_FN.format(site=config.SITE_SHORTNAME[site_info['sitecode']], 
                                                   span=site_info['span'],
                                                   segment=i)
        output_dir = os.path.join('..', 'data', 'processed')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        output_path = os.path.join(output_dir, output_fn)
        df.to_csv(output_path)
        print(f'Wrote segment {i} to {output_path}')
        
def read_vented_segments(sitecode):
    vented_segments = []

    vented_dir_path = os.path.join('..', '..', 'vented', 'data', 'processed')
    for fn in os.listdir(vented_dir_path):
        if config.SITE_SHORTNAME[sitecode] in fn:
            data_path = os.path.join(vented_dir_path, fn)
            print(f'loading {data_path}')
            vented_segments.append(pd.read_csv(data_path, index_col=0, parse_dates=[0], infer_datetime_format=True))
    return vented_segments

def read_unvented_segments(sitecode, start_year, end_year):
    unvented_segments = {}
    for year in range(start_year, end_year + 1):
        unvented_segments[year] = []
        year_dir = os.path.join('..', '..', f'unvented_{year}')
        unvented_data_dir = os.path.join(year_dir, 'data', 'barocorrected')
        for fn in sorted(os.listdir(unvented_data_dir)):
            if config.SITE_SHORTNAME[sitecode] in fn:
                data_path = os.path.join(unvented_data_dir, fn)
                print(f'loading {data_path}')
                unvented_segments[year].append(pd.read_csv(data_path, 
                                                           index_col=0, 
                                                           parse_dates=[0], 
                                                           infer_datetime_format=True))
    return unvented_segments

def plot_all(title=None, prev=None, prev_weeks=24, stage=None, vented_segments=None, unvented_segments=None, vented_offsets=None, unvented_offsets=None, figsize=None):
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if prev is not None:
        td = timedelta(weeks=prev_weeks)
        end_dt = prev.index[-1]
        ax.plot(prev[end_dt - td:], label='previous record', color='k')
    
    if vented_segments is not None:
        if vented_offsets is None:
            vented_offsets = np.zeros(len(vented_segments))
        for i, (segment, offset) in enumerate(zip(vented_segments, vented_offsets)):
            ax.plot(segment['vented_pressure(cm)'] + offset, label=f'vented, segment {i}')
  
    if unvented_segments is not None:
        for year in unvented_segments:
            unvented_segment_offsets = []
            if unvented_offsets is None:
                unvented_segment_offsets = np.zeros(len(unvented_segments[year]))
            else:
                unvented_segment_offsets = unvented_offsets[year]
            for i, (segment, offset) in enumerate(zip(unvented_segments[year], unvented_segment_offsets)):
                ax.plot(segment['barocorrected_pressure(cm)'] + offset, label=f'unvented {year}, segment {i}')
    
    if stage is not None:
        ax.scatter(stage.index, stage['stage(cm)'], label='stage', color='k')
        
    plt.title(title)
    plt.xticks(rotation=30)
    plt.legend()
    
# return a list of the diference bewtween each overlapping time stamp
def dif_btw_series(ds1, ds2):
    # find overlapping time stamps
    dt_intersection = ds1.index.intersection(ds2.index)
    
    # compute difference, store as list
    return ds1[dt_intersection] - ds2[dt_intersection]

# returns list of the difference between each stage measurement and an interpolated sample from the time series
def dif_btw_stage_series(stage_ds, logger_ds):
    dt_range = stage_ds.index[(stage_ds.index >= logger_ds.index[0]) & (stage_ds.index <= logger_ds.index[-1])]
    
    series_interp = pd.Series(np.interp(dt_range.asi8, logger_ds.index.asi8, logger_ds))
    series_interp = series_interp.set_axis(dt_range)
    return stage_ds - series_interp

def initialize_offsets(vented_segments, unvented_segments):
    
    vented_offsets = None if vented_segments is None else np.zeros(len(vented_segments))

    unvented_offsets = {}
    for year in unvented_segments:
        unvented_offsets[year] = np.zeros(len(unvented_segments[year]))
        
    return vented_offsets, unvented_offsets

def stitch_timeseries(segments):    
    # add each segment to dataframe s.t. the added segment goes from the specified start to the
    # specified start of the next segment
    trimmed_segments=[]
    for i in range(len(segments)):
        df = segments[i][0]
        start = segments[i][1]
        end = segments[i+1][1] if (i < len(segments) - 2) else segments[i][0].index[-1]
        offset = segments[i][2]

        # change vented header df header to match unvented
        if 'vented_pressure(cm)' in df:
            df['barocorrected_pressure(cm)'] = df['vented_pressure(cm)']
            df = df.rename(columns={'vented_pressure(cm)':'raw_pressure(cm)'})
        
        segment = df[start:end]
        offset_series = segment['barocorrected_pressure(cm)'] + offset
        segment['adjusted_stage(cm)'] = offset_series

        trimmed_segments.append(segment)
    result = pd.concat(trimmed_segments)
    
    # add empty columns for raw, barocorrected, and/or vented if not present
    #for label in ['raw_pressure(cm)', 'barocorrected_pressure(cm)', 'vented_pressure(cm)']:
        #if label not in result:
            #result[label] = np.nan
        
    # add column for estimated discharge
    result['estimated_discharge(cms)'] = np.nan
    result = result[config.FINAL_OUTPUT_HEADER[1:]]
    
    return result

def compute_discharge(h, sitecode):
    return (h / 100.0).apply(config.RATING_CURVES[sitecode]).apply(lambda x: x.real)

def plot_discharge(stitched_df, sitecode):
    plt.figure()
    plt.plot(stitched_df['estimated_discharge(cms)'], label='Full Series')
    plt.plot(stitched_df[stitched_df['discharge_flag'] == 2]['estimated_discharge(cms)'], label='Ice Jam')
    plt.plot(stitched_df[stitched_df['discharge_flag'] == 3]['estimated_discharge(cms)'], label='Malfunction')
    plt.xticks(rotation=30)
    plt.ylabel('cms')
    plt.title(f'Estimated Discharge for {config.SITE_LONGNAME[sitecode]}')
    plt.legend()
    
def save_final_data(df, sitecode, startyear, endyear):
    # format file name
    output_fn = config.FINAL_OUTPUT_FN.format(site=config.SITE_SHORTNAME[sitecode],
                                              start=startyear,
                                              end=endyear)
    
    # make directory for processed data
    output_dir = os.path.join('..', 'data', 'processed')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # save data to processed directory
    output_path = os.path.join(output_dir, output_fn)
    df.to_csv(output_path)
    
    print(f'Wrote data to {output_path}')
    
def choose_column_dtype(df, dtypes):
    for column in dtypes:
        df.iloc[:, column] = df.iloc[:, column].astype(dtypes[column])
    return df

def map_old_labels_2_new(old_df, old_labels):
    columns = [str(l) for l in old_labels if l != '']
    name_map = {old:new for old, new in zip(old_labels, config.FINAL_OUTPUT_HEADER[1:]) if old != ''}
    return old_df[columns].rename(columns=name_map)

def join_dataframes(df1, df2):
    intersection = df1.index.intersection(df2.index)
    new_start = intersection[-1] + pd.Timedelta(minutes=30) if len(intersection) != 0 else df2.index[0]
    
    print('Joining old and new series')
    print(f'End of old series: {df1.index[-1]}')
    print(f'Start of new series: {new_start}')
     
    # concat
    return pd.concat([df1, df2[new_start:]])[config.FINAL_OUTPUT_HEADER[1:]], df1.index[-1]

def plot_boundary(df, boundary):
    start = boundary - timedelta(minutes=60)
    stop = boundary + timedelta(minutes=60)

    fig, ax = plt.subplots(1, 2)
    df['adjusted_stage(cm)'][start:stop].plot(ax=ax[0])
    ax[0].set_ylabel('cm')


    df['estimated_discharge(cms)'][start:stop].plot(ax=ax[1])
    ax[1].set_ylabel('cms')
    ax[1].set_title('estimated discharge')

    return df[start:stop].head()
import pandas as pd
import numpy as np
import os
import sys


def log_interp(df=None, step=0.125, log_col_name=None, log_col_name_output=None, depth_col_name='Depth',
               top_col_name=None, bot_col_name=None, nominal=True):
    """
    Interpolate well log between top depth and bottom depth.
    :param df: (pandas.DataFrame) - Well log data frame which contains ['top depth', 'bottom depth', 'log'] column.
    :param step: (Float) - Default is 0.125m. Interpolation interval.
    :param log_col_name: (String) - Column name of the well log.
    :param log_col_name_output: (String) - Default is log_col_name. Column name of the interpolated well log.
    :param depth_col_name: (String) - Default is 'Depth'. Column name of measured depth.
    :param top_col_name: (String) - Column name of the top boundary depth.
    :param bot_col_name: (String) - Column name of the bottom boundary depth.
    :param nominal: (Bool) - Default is True. Whether the log value is nominal.
    :return df_out: (pandas.DataFrame) - Interpolated well log data frame.
    """
    depth_min = df[top_col_name].min()  # Get minimum depth.
    depth_max = df[bot_col_name].max()  # Get maximum depth.
    depth_array = np.arange(depth_min, depth_max + step, step, dtype=np.float32)  # Create depth column.
    log_array = np.full(len(depth_array), np.nan)  # Initiate well log column with NaN.
    df_out = pd.DataFrame({depth_col_name: depth_array, log_col_name_output: log_array})  # Create new data frame.
    # Assign log values to new data frame.
    for i in range(len(df)):
        idx = (df_out.loc[:, depth_col_name] <= df.loc[i, bot_col_name]) & \
                (df_out.loc[:, depth_col_name] >= df.loc[i, top_col_name])
        df_out.loc[idx, log_col_name_output] = df.loc[i, log_col_name]
    # Delete rows with NaN.
    df_out.dropna(axis='index', how='any', inplace=True)
    # Change data type.
    if nominal:
        df_out[log_col_name_output] = df_out[log_col_name_output].astype('int32')
    else:
        df_out[log_col_name_output] = df_out[log_col_name_output].astype('float32')
    return df_out


def time_log(df_dt=None, df_log=None, depth_col='Depth', time_col='TWT', log_col=None, fillna=-999, nominal=False):
    """
    Match well logs with a detailed time-depth relation.
    :param df_dt: (pandas.DataFrame) - Time-depth relation data frame which contains ['Depth', 'Time'] columns.
    :param df_log: (pandas.DataFrame) - Well log data frame which contains ['Depth', 'log'] columns.
    :param depth_col: (String) - Default is 'Depth'. Column name of depth.
    :param time_col: (String) - Default is 'TWT'. Column name of two-way time.
    :param log_col: (String) - Column name of well log.
    :param fillna: (Float or integer) - Fill NaN with this number.
    :param nominal: (Bool) - Default is False. Whether the log value is nominal.
    :return: df_out: (pandas.DataFrame) - Time domain well log data frame.
    """
    # Set flag to speed up search.
    flag = 0
    # Auto-break indicator.
    autobreak = 0
    for i in range(len(df_dt)):
        if df_dt.loc[i, depth_col] < np.amin(df_log[depth_col].values):
            continue
        elif df_dt.loc[i, depth_col] > np.amax(df_log[depth_col].values):
            print('\n\t\tDT depth range is out of log depth range. Process finished.')
            autobreak = 1
            break
        for j in range(flag, len(df_log) - 1, 1):
            top_log_depth = df_log.loc[j, depth_col]
            bottom_log_depth = df_log.loc[j + 1, depth_col]
            if top_log_depth <= df_dt.loc[i, depth_col] <= bottom_log_depth:
                df_dt.loc[i, log_col] = df_log.loc[j, log_col]
                flag = j
                break
        sys.stdout.write(
            '\r\t\tProgress: %d/%d samples %.2f%%' % (i + 1, len(df_dt), (i + 1) / len(df_dt) * 100))  # 打印进程
        if i == len(df_dt) - 1:
            sys.stdout.write('\n')
    if not autobreak:
        print('\n\t\tProcess finished.')
    df_out = df_dt[[time_col, log_col]].copy()
    df_out.fillna(fillna, inplace=True)
    if nominal:
        df_out[log_col] = df_out[log_col].astype('int32')
    else:
        df_out[log_col] = df_out[log_col].astype('float32')
    return df_out


if __name__ == '__main__':
    # Set directories and file names.
    root_dir = 'D:/Opendtect/Database/Niuzhuang/'
    log_dir = 'Well logs/LithoCodeForPetrel-interpolated'
    dt_dir = 'Well logs TD (new)'
    output_dir = 'Well logs/LithoCodeForPetrel-time'

    # Depth-time file list.
    dt_file_list = os.listdir(os.path.join(root_dir, dt_dir))

    # Log file list.
    log_file_list = os.listdir(os.path.join(root_dir, log_dir))

    for dt_file in dt_file_list:
        # Get well name from depth-time file.
        well_name = dt_file[:-4]
        print('Processing well %s' % well_name)
        # Read depth-time file.
        df_dt = pd.read_csv(os.path.join(root_dir + dt_dir, dt_file), delimiter='\t')
        # Print control.
        file_match = 0
        for log_file in log_file_list:
            # Find the corresponding well in log files.
            if well_name in log_file:
                file_match = 1
                print('\tCorresponding log file: %s' % log_file)
                # Read log file.
                df_log = pd.read_csv(os.path.join(root_dir + log_dir, log_file), delimiter='\t')
                df_out = time_log(df_dt, df_log, log_col='Litho_Code', nominal=True)
                df_out.to_csv(os.path.join(root_dir + output_dir, dt_file[:-4] + '.txt'), sep='\t', index=False)
                break
        if file_match == 0:
            print('\tNo corresponding log file.')

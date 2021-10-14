import pandas as pd
import numpy as np
import os
import sys


def resample_log(df_log=None, delta=None, depth_col='depth', log_col=None, method='average', abnormal_value=None,
                 nominal=False):
    """
    Re-sample well log by a certain depth interval.
    :param df_log: (pandas.DataFrame) - Well log data frame which contains ['depth', 'log'] columns.
    :param delta: (Float) - Depth interval.
    :param depth_col: (String) - Default is 'depth'. Depth column name.
    :param log_col: (String or list of strings) - Well log column name(s). E.g. 'gamma' or ['Vp', 'porosity'].
    :param method: (String) - Default is 'average'. Re-sampling method.
                   'nearest' - Take the nearest log value.
                   'average' - Take the mean of log values in a depth window (length = delta)
                               centered at the re-sampled depth point.
                   'median' - Take the median of log values in a depth window (length = delta)
                              centered at the re-sampled depth point.
                   'rms' - Take the root-mean-square of log values in a depth window (length = delta)
                           centered at the re-sampled depth point.
                   'most_frequent' - Take the most frequent log values in a depth window (length = delta)
                                     centered at the re-sampled depth point.
    :param abnormal_value: (Float) - Abnormal value in log column. If abnormal value is defined, will remove the whole
                                     row with abnormal value. If None, means no abnormal value in log column.
    :param nominal: (Bool) - Default is False. Whether the log value is nominal (e.g. [0, 1, 2, 0, 2]). If True, the log
                    value will be integer, else the log value will be float.
    :return: df_out: (pandas.Dataframe) - Re-sampled well log data frame.
    """
    if abnormal_value is not None:
        df_log.replace(abnormal_value, np.nan)
        df_log.dropna(axis='index', how='any', inplace=True)
        df_log.reset_index(drop=True, inplace=True)
    depth = df_log[depth_col].values  # Original depth array.
    log = df_log[log_col].values  # Original log array.
    new_depth = np.arange(start=np.amin(depth) // delta * delta,
                          stop=np.amax(depth) // delta * delta + delta * 2,
                          step=delta)  # New depth array.
    if log.ndim > 1:
        new_log = np.full([len(new_depth), log.shape[1]], fill_value=np.nan)  # Initiate new log array (fill with nan).
    else:
        new_log = np.full(len(new_depth), fill_value=np.nan)
    for i in range(len(new_depth)):
        # Choose the depth and log values that fit the condition.
        if new_depth[i] == np.amin(new_depth):  # Start point of new depth.
            condition = (depth > new_depth[i]) & (depth <= new_depth[i] + delta / 2)
        elif new_depth[i] == np.amax(new_depth):  # End point of new depth.
            condition = (depth > new_depth[i] - delta / 2) & (depth <= new_depth[i])
        else:  # Inner points of new depth.
            condition = (depth > new_depth[i] - delta / 2) & (depth <= new_depth[i] + delta / 2)
        index = np.argwhere(condition)  # Find index in the window.
        temp_log = log[index]  # Log in the window.
        temp_depth = depth[index]  # Depth in the window.
        # Re-sample log by different methods.
        if method == 'nearest' and len(temp_log):  # Nearest neighbor.
            ind_nn = np.argmin(np.abs(temp_depth - new_depth[i]))  # The nearest neighbor index.
            new_log[i] = temp_log[ind_nn]
        if method == 'average' and len(temp_log):  # Take average value.
            new_log[i] = np.average(temp_log, axis=0)
        if method == 'median' and len(temp_log):  # Take median value.
            new_log[i] = np.median(temp_log, axis=0)
        if method == 'rms' and len(temp_log):  # Root-mean-square value.
            new_log[i] = np.sqrt(np.mean(temp_log ** 2, axis=0))
        if method == 'most_frequent' and len(temp_log):  # Choose the most frequent log value (for nominal log only).
            if temp_log.ndim > 1 and temp_log.shape[1] > 1:
                for j in range(temp_log.shape[1]):
                    values, counts = np.unique(temp_log[:, j], return_counts=True)
                    ind_mf = np.argmax(counts)
                    new_log[i, j] = values[ind_mf]
            else:
                values, counts = np.unique(temp_log, return_counts=True)
                ind_mf = np.argmax(counts)
                new_log[i] = values[ind_mf]
    # Output result to new data-frame.
    df_out = pd.DataFrame(data=np.c_[new_depth, new_log], columns=df_log.columns)
    df_out.dropna(axis='index', how='any', inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    # Change data type.
    df_out = df_out.astype('float32')
    if nominal:
        df_out[log_col] = df_out[log_col].astype('int32')
    return df_out


def log_interp(df=None, step=0.125, log_col=None, depth_col='Depth',
               top_col=None, bottom_col=None, nominal=True):
    """
    Interpolate well log between top depth and bottom depth.
    :param df: (pandas.DataFrame) - Well log data frame which contains ['top depth', 'bottom depth', 'log'] column.
    :param step: (Float) - Default is 0.125m. Interpolation interval.
    :param log_col: (String or list of strings) - Column name(s) of the well log.
    :param depth_col: (String) - Default is 'Depth'. Column name of measured depth.
    :param top_col: (String) - Column name of the top boundary depth.
    :param bottom_col: (String) - Column name of the bottom boundary depth.
    :param nominal: (Bool) - Default is True. Whether the log value is nominal.
    :return df_out: (pandas.DataFrame) - Interpolated well log data frame.
    """
    depth_min = df[top_col].min()  # Get minimum depth.
    depth_max = df[bottom_col].max()  # Get maximum depth.
    depth_array = np.arange(depth_min, depth_max + step, step, dtype=np.float32)  # Create depth column.
    # Initiate well log column with NaN.
    if isinstance(log_col, str):
        log_array = np.full(len(depth_array), np.nan)
    elif isinstance(log_col, list) and len(log_col) > 1:
        log_array = np.full([len(depth_array), len(log_col)], np.nan)
    else:
        raise ValueError('Log column name must either be string type for 1 column or list type for 2 or more columns.')
    # Create new data frame.
    df_out = pd.DataFrame({depth_col: depth_array})
    df_out[log_col] = log_array
    # Assign log values to new data frame.
    for i in range(len(df)):
        idx = (df_out.loc[:, depth_col] <= df.loc[i, bottom_col]) & \
                (df_out.loc[:, depth_col] >= df.loc[i, top_col])
        df_out.loc[idx, log_col] = df.loc[i, log_col]
    # Delete rows with NaN.
    df_out.dropna(axis='index', how='any', inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    # Change data type.
    df_out = df_out.astype('float32')
    if nominal:
        df_out[log_col] = df_out[log_col].astype('int32')
    return df_out


def time_log(df_dt=None, df_log=None, log_depth_col='Depth', dt_depth_col='Depth',
             time_col='TWT', log_col=None, fillna=-999, nominal=False):
    """
    Match well logs with a detailed time-depth relation.
    :param df_dt: (pandas.DataFrame) - Depth-time relation data frame which contains ['Depth', 'Time'] columns.
    :param df_log: (pandas.DataFrame) - Well log data frame which contains ['Depth', 'log'] columns.
    :param log_depth_col: (String) - Default is 'Depth'. Column name of depth in well log file.
    :param dt_depth_col: (String) - Default is 'Depth'. Column name of depth in depth-time relation file.
    :param time_col: (String) - Default is 'TWT'. Column name of two-way time.
    :param log_col: (String or list of strings) - Column name(s) of well log.
    :param fillna: (Float or integer) - Fill NaN with this number.
    :param nominal: (Bool) - Default is False. Whether the log value is nominal.
    :return: df_out: (pandas.DataFrame) - Time domain well log data frame.
    """
    # Set flag to speed up search.
    flag = 0
    for i in range(len(df_dt)):
        if df_dt.loc[i, dt_depth_col] < np.amin(df_log[log_depth_col].values):
            continue
        elif df_dt.loc[i, dt_depth_col] > np.amax(df_log[log_depth_col].values):
            print('D-T relation depth range is out of well log depth range. Auto-break.')
            print('Process finished.')
            break
        for j in range(flag, len(df_log) - 1, 1):
            top_log_depth = df_log.loc[j, log_depth_col]
            bottom_log_depth = df_log.loc[j + 1, log_depth_col]
            if top_log_depth <= df_dt.loc[i, dt_depth_col] <= bottom_log_depth:
                df_dt.loc[i, log_col] = df_log.loc[j, log_col]
                flag = j
                break
        # Print progress.
        sys.stdout.write('\rProgress: %.2f%% [%d/%d samples]' % ((i+1)/len(df_dt) * 100, i+1, len(df_dt)))
    sys.stdout.write('\n')
    if isinstance(log_col, str):
        df_out = df_dt[[time_col, log_col]].copy()
    elif isinstance(log_col, list) and len(log_col) > 1:
        df_out = df_dt[[time_col] + log_col].copy()
    else:
        raise ValueError('Log column name must either be string type for 1 column or list type for 2 or more columns.')
    df_out.fillna(fillna, inplace=True)
    # Change data type.
    df_out = df_out.astype('float32')
    if nominal:
        df_out[log_col] = df_out[log_col].astype('int32')
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

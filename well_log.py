import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import sys
import math


def resample_log(df_log=None, delta=None, depth_col='depth', log_col=None, method='average', abnormal_value=None,
                 nominal=False):
    """
    Re-sample well log by a certain depth interval (Small sampling interval to large sampling interval).
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
    new_depth = np.arange(start=math.ceil(np.amin(depth) // delta * delta),
                          stop=np.amax(depth) // delta * delta + delta * 2,
                          step=delta)  # New depth array.
    if log.ndim > 1:
        new_log = np.full([len(new_depth), log.shape[1]], fill_value=np.nan)  # Initiate new log array (fill with nan).
    else:
        new_log = np.full(len(new_depth), fill_value=np.nan)
    for i in range(len(new_depth)):
        # Choose the depth and log values that fit the condition.
        if new_depth[i] == np.amin(new_depth):  # Start point of new depth.
            condition = (depth >= new_depth[i]) & (depth <= new_depth[i] + delta / 2)
        elif new_depth[i] == np.amax(new_depth):  # End point of new depth.
            condition = (depth > new_depth[i] - delta / 2) & (depth <= new_depth[i])
        else:  # Inner points of new depth.
            condition = (depth > new_depth[i] - delta / 2) & (depth <= new_depth[i] + delta / 2)
        index = np.argwhere(condition)  # Find index in the window.
        temp_log = log[index]  # Log in the window.
        temp_depth = depth[index]  # Depth in the window.
        if len(temp_log):  # If there are log values in the window.
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
            if method == 'most_frequent' and len(temp_log):  # Choose the most frequent log value (nominal log only).
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
               top_col=None, bottom_col=None, nominal=True, mode='segmented', method='slinear'):
    """
    Interpolate well log between top depth and bottom depth.
    :param df: (pandas.DataFrame) - Well log data frame which contains ['top depth', 'bottom depth', 'log'] column.
    :param step: (Float) - Default is 0.125m. Interpolation interval.
    :param log_col: (String or list of strings) - Column name(s) of the well log.
    :param depth_col: (String) - Default is 'Depth'. Column name of measured depth.
    :param top_col: (String) - Column name of the top boundary depth.
    :param bottom_col: (String) - Column name of the bottom boundary depth.
    :param nominal: (Bool) - Default is True. Whether the log value is nominal.
    :param mode: (String) - Default is 'segmented'. When is 'segmented', interpolate segmented well log with top depth
                 and bottom depth column. When is 'continuous', interpolate continuous well log with depth column.
    :param method: (String) - Default is 'slinear'. Interpolation method when processing continuous well log values.
                   Optional methods are: 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic',
                   'previous', and 'next'. 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of
                   zeroth, first, second or third order; 'previous' and 'next' simply return the previous or next value
                   of the point; 'nearest-up' and 'nearest' differ when interpolating half-integers (e.g. 0.5, 1.5) in
                   that 'nearest-up' rounds up and 'nearest' rounds down.
    :return df_out: (pandas.DataFrame) - Interpolated well log data frame.
    """
    if mode == 'segmented':
        depth_min = df[top_col].min()  # Get minimum depth.
        depth_min = math.ceil(depth_min)
        depth_max = df[bottom_col].max()  # Get maximum depth.
        depth_array = np.arange(depth_min, depth_max + step, step, dtype=np.float32)  # Create depth column.
        # Initiate well log column with NaN.
        if isinstance(log_col, str):
            log_array = np.full(len(depth_array), np.nan)
        elif isinstance(log_col, list) and len(log_col) > 1:
            log_array = np.full([len(depth_array), len(log_col)], np.nan)
        else:
            raise ValueError('Log column name must either be string type for 1 column or list type for 2 or more '
                             'columns.')
        # Create new data frame.
        df_out = pd.DataFrame({depth_col: depth_array})
        df_out[log_col] = log_array
        # Assign log values to new data frame.
        for i in range(len(df)):
            idx = (df_out.loc[:, depth_col] <= df.loc[i, bottom_col]) & \
                    (df_out.loc[:, depth_col] >= df.loc[i, top_col])
            df_out.loc[idx, log_col] = df.loc[i, log_col]
    elif mode == 'continuous':
        log_depth = df[depth_col].values  # Get well log depth array.
        log_value = df[log_col].values  # Get well log value array.
        depth_min = np.amin(log_depth)  # Get minimum well log value.
        depth_max = np.amax(log_depth)  # Get maximum well log value.
        log_depth[log_depth == depth_min] = round(depth_min)
        new_depth = np.arange(start=depth_min, stop=depth_max, step=step)  # New depth array.
        f = scipy.interpolate.interp1d(log_depth, log_value, axis=0, kind=method)  # Interpolator.
        new_value = f(new_depth)  # Interpolated log value.
        data = np.c_[new_depth, new_value]
        df_out = pd.DataFrame(data, columns=df.columns)
    else:
        raise ValueError("Mode can only be 'segmented' or 'continuous'.")
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


def cross_plot2D(df=None, x=None, y=None, c=None, cmap='rainbow',
                 xlabel=None, ylabel=None, title=None, colorbar=None,
                 xlim=None, ylim=None, show=True):
    plt.figure()
    # Set style sheet to bmh.
    plt.style.use('bmh')
    # Set up the scatter plot.
    plt.scatter(x=x, y=y, data=df, c=c, cmap=cmap)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.title(title, fontsize=16)
    cbar = plt.colorbar()
    cbar.set_label(colorbar, size=14)
    if show:
        plt.show()


def crossplot3D(df=None, x=None, y=None, z=None, c=None, cmap='rainbow',
                xlabel=None, ylabel=None, zlabel=None, title=None, colorbar=None,
                xlim=None, ylim=None, zlim=None, show=True):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # Get values from data frame.
    xs = df[x].values
    ys = df[y].values
    zs = df[z].values
    cs = df[c].values
    # Set up the scatter plot.
    scat = ax.scatter3D(xs, ys, zs, c=cs, cmap=cmap)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_zlabel(zlabel, fontsize=14)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if zlim is not None:
        ax.set_zlim(zlim[0], zlim[1])
    plt.title(title, fontsize=16)
    cbar = fig.colorbar(scat, ax=ax)
    cbar.set_label(colorbar, size=16)
    if show:
        plt.show()

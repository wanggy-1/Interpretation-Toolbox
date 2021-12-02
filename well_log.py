import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.spatial
import math
import segyio
from prettytable import PrettyTable


def resample_log(df_log=None, delta=None, depth_col='depth', log_col=None, method='average', abnormal_value=None,
                 nominal=False, delete_nan=True, fill_nan=None):
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
    :param delete_nan: (Bool) - Default is True. Whether to delete rows with NaN value.
    :param fill_nan: (Integer or float) - Default is not to fill NaN. Fill NaN with this value.
    :return: df_out: (pandas.Dataframe) - Re-sampled well log data frame.
    """
    df_log_copy = df_log.copy()
    if abnormal_value is not None:
        df_log_copy.replace(abnormal_value, np.nan)
        df_log_copy.dropna(axis='index', how='any', inplace=True)
        df_log_copy.reset_index(drop=True, inplace=True)
    depth = df_log_copy[depth_col].values  # Original depth array.
    log = df_log_copy[log_col].values  # Original log array.
    new_depth = np.arange(start=math.ceil(np.amin(depth) // delta * delta),
                          stop=np.amax(depth) // delta * delta + delta * 2,
                          step=delta)  # New depth array.
    if log.ndim > 1:
        new_log = np.full([len(new_depth), log.shape[1]], fill_value=np.nan)  # Initiate new log array (fill with nan).
    else:
        new_log = np.full(len(new_depth), fill_value=np.nan)
    for i in range(len(new_depth)):
        # Choose the depth and log values that meet the condition.
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
    df_out = pd.DataFrame(data=np.c_[new_depth, new_log], columns=df_log_copy.columns)
    # Remove rows with all missing log values (NaN).
    sub_col = list(df_out.columns)
    sub_col.remove(depth_col)
    df_out.dropna(axis='index', how='all', subset=sub_col, inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    if fill_nan:
        # Fill NaN.
        df_out.fillna(value=fill_nan, inplace=True)
    if delete_nan:
        # Delete rows with any NaN.
        df_out.dropna(axis='index', how='any', inplace=True)
        df_out.reset_index(drop=True, inplace=True)
    # Change data type.
    df_out = df_out.astype('float32')
    if nominal:
        df_out[log_col] = df_out[log_col].astype('int32')
    return df_out


def log_interp(df=None, step=0.125, log_col=None, depth_col='Depth',
               top_col=None, bottom_col=None, nominal=True, mode='segmented', method='slinear',
               delete_nan=False, fill_nan=None):
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
    :param delete_nan: (Bool) - Default is True. Whether to delete rows with NaN.
    :param fill_nan: (Integer or float) - Default is not to fill NaN. Fil NaN with this value.
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
    # Remove rows with all missing log values (NaN).
    sub_col = list(df_out.columns)
    sub_col.remove(depth_col)
    df_out.dropna(axis='index', how='all', subset=sub_col, inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    if fill_nan is not None:
        # Fill NaN.
        df_out.fillna(value=fill_nan, inplace=True)
    if delete_nan:
        # Delete rows with NaN.
        df_out.dropna(axis='index', how='any', inplace=True)
        df_out.reset_index(drop=True, inplace=True)
    # Change data type.
    df_out = df_out.astype('float32')
    if nominal:
        df_out[log_col] = df_out[log_col].astype('int32')
    return df_out


def time_log(df_dt=None, df_log=None, log_depth_col='Depth', dt_depth_col='Depth',
             time_col='TWT', log_col=None, fill_nan=None, delete_nan=False, nominal=False):
    """
    Match well logs with a detailed time-depth relation.
    :param df_dt: (pandas.DataFrame) - Depth-time relation data frame which contains ['Depth', 'Time'] columns.
    :param df_log: (pandas.DataFrame) - Well log data frame which contains ['Depth', 'log'] columns.
    :param log_depth_col: (String) - Default is 'Depth'. Column name of depth in well log file.
    :param dt_depth_col: (String) - Default is 'Depth'. Column name of depth in depth-time relation file.
    :param time_col: (String) - Default is 'TWT'. Column name of two-way time.
    :param log_col: (String or list of strings) - Column name(s) of well log.
    :param fill_nan: (Float or integer) - Default is not to fill NaN. Fill NaN with this number.
    :param delete_nan: (Bool) - Default is False. Whether to delete rows with NaN.
    :param nominal: (Bool) - Default is False. Whether the log value is nominal.
    :return: df_log_time: (pandas.DataFrame) - Time domain well log data frame.
    """
    df_log_time = df_log.copy()
    if log_col == 'infer':
        # Infer log columns.
        log_col = list(df_log_time.columns).remove(log_depth_col)
    # Remove rows in well log whose depth is out of range of depth-time relation.
    ind = [i for i in range(len(df_log_time)) if df_log_time.loc[i, log_depth_col] < df_dt[dt_depth_col].min() or
           df_log_time.loc[i, log_depth_col] > df_dt[dt_depth_col].max()]
    df_log_time.drop(ind, inplace=True)
    df_log_time.reset_index(drop=True, inplace=True)
    # Get depth and time arrays in df_dt.
    depth = df_dt[dt_depth_col].values
    time = df_dt[time_col].values
    # Fit interpolator.
    f = scipy.interpolate.interp1d(depth, time)
    # Transform depth-domain well log to time-domain.
    df_log_time[time_col] = f(df_log_time[log_depth_col].values)
    # Drop depth column.
    df_log_time.drop(columns=log_depth_col, inplace=True)
    # Re-arrange columns in df_log_copy.
    if isinstance(log_col, str):
        log_col = [log_col]
    new_col = [time_col] + log_col
    df_log_time = df_log_time[new_col]
    # Remove rows with all missing log values (NaN).
    sub_col = list(df_log_time.columns)
    sub_col.remove(time_col)
    df_log_time.dropna(axis='index', how='all', subset=sub_col, inplace=True)
    df_log_time.reset_index(drop=True, inplace=True)
    if fill_nan is not None:
        # Fill NaN.
        df_log_time.fillna(value=fill_nan, inplace=True)
    if delete_nan:
        # Delete rows with NaN.
        df_log_time.dropna(axis='index', how='any', inplace=True)
        df_log_time.reset_index(drop=True, inplace=True)
    # Change data type.
    df_log_time = df_log_time.astype('float32')
    if nominal:
        df_log_time[log_col] = df_log_time[log_col].astype('int32')
    return df_log_time


def cross_plot2D(df=None, x=None, y=None, c=None, cmap='rainbow',
                 xlabel=None, ylabel=None, title=None, colorbar=None,
                 xlim=None, ylim=None, show=True):
    """
    Make 2D cross-plot.
    https://towardsdatascience.com/scatterplot-creation-and-visualisation-with-matplotlib-in-python-7bca2a4fa7cf
    :param df: (pandas.DataFrame) - Well log data frame.
    :param x: (String) - Log name in data frame, which will be the x-axis of the cross-plot.
    :param y: (String) - Log name in data frame, which will be the y-axis of the cross-plot.
    :param c: (String) - Log name in data frame, which will be the color of the scatters in cross-plot.
    :param cmap: (String) - Default is 'rainbow', the color map of scatters.
    :param xlabel: (String) - X-axis name.
    :param ylabel: (String) - Y-axis name.
    :param title: (String) - Title of the figure.
    :param colorbar: (String) - Name of the color-bar.
    :param xlim: (List of floats) - Default is to infer from data. Range of x-axis, e.g. [0, 150]
    :param ylim: (List of floats) - Default is to infer from data. Range of y-axis, e.g. [0, 150]
    :param show: (Bool) - Default is True. Whether to show the figure.
    """
    plt.figure(figsize=(13, 9))
    # Set style sheet to bmh.
    plt.style.use('bmh')
    # Set up the scatter plot.
    plt.scatter(x=x, y=y, data=df, c=c, cmap=cmap)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.tick_params(labelsize=16)
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.title(title, fontsize=20)
    cbar = plt.colorbar()
    cbar.set_label(colorbar, size=18)
    cbar.ax.tick_params(labelsize=16)
    if show:
        plt.show()


def cross_plot3D(df=None, x=None, y=None, z=None, c=None, cmap='rainbow',
                 xlabel=None, ylabel=None, zlabel=None, title=None, colorbar=None,
                 xlim=None, ylim=None, zlim=None, show=True):
    """
    Make 3D cross-plot.
    :param df: (pandas.DataFrame) - Well log data frame.
    :param x: (String) - Log name in data frame, which will be the x-axis of the cross-plot.
    :param y: (String) - Log name in data frame, which will be the y-axis of the cross-plot.
    :param z: (String) - Log name in data frame, which will be the z-axis of the cross-plot.
    :param c: (String) - Log name in data frame, which will be the color of the scatters in cross-plot.
    :param cmap: (String) - Default is 'rainbow', the color map of scatters.
    :param xlabel: (String) - X-axis name.
    :param ylabel: (String) - Y-axis name.
    :param zlabel: (String) - Z-axis name.
    :param title: (String) - Title of the figure.
    :param colorbar: (String) - Name of the color-bar.
    :param xlim: (List of floats) - Default is to infer from data. Range of x-axis, e.g. [0, 150]
    :param ylim: (List of floats) - Default is to infer from data. Range of y-axis, e.g. [0, 150]
    :param zlim: (List of floats) - Default is to infer from data. Range of z-axis, e.g. [0, 150]
    :param show: (Bool) - Default is True. Whether to show the figure.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = plt.axes(projection='3d')
    # Get values from data frame.
    xs = df[x].values
    ys = df[y].values
    zs = df[z].values
    cs = df[c].values
    # Set up the scatter plot.
    scat = ax.scatter3D(xs, ys, zs, c=cs, cmap=cmap)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_zlabel(zlabel, fontsize=18)
    ax.tick_params(labelsize=14)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if zlim is not None:
        ax.set_zlim(zlim[0], zlim[1])
    plt.title(title, fontsize=20)
    cbar = fig.colorbar(scat, ax=ax)
    cbar.set_label(colorbar, size=18)
    cbar.ax.tick_params(labelsize=16)
    if show:
        plt.show()


def plotlog(df=None, depth=None, log=None, fill_log=True, cmap='rainbow',
            xlabel=None, ylabel='Depth - m', xlim=None, ylim=None,
            title=None, show=True):
    """
    Draw well log curves.
    https://towardsdatascience.com/enhancing-visualization-of-well-logs-with-plot-fills-72d9dcd10c1b
    :param df: (pandas.DataFrame) - Well log data frame.
    :param depth: (String) - Depth column name in data frame.
    :param log: (String) - Log column name in data frame.
    :param fill_log: (Bool) - Default is True which is to fill the area under the log curve.
    :param cmap: (String) - Default is 'rainbow'. Color map to fill the area under the log curve.
    :param xlabel: (String) - X-axis name.
    :param ylabel: (String) - Default is 'Depth - m'. Y-axis name.
    :param xlim: (List of floats) - Default is to infer from data. Range of x-axis, e.g. [0, 150].
    :param ylim: (List of floats) - Default is to infer from data. Range of y-axis, e.g. [0, 2000].
    :param title: (String) - Title of the figure.
    :param show: (Bool) - Default is True. Whether to show the figure.
    """
    # Set up the plot.
    plt.style.use('bmh')  # PLot style.
    plt.figure(figsize=(7, 10))
    ax = plt.axes()
    plt.plot(log, depth, data=df, c='black', lw=0.5)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is None:
        ax.set_ylim(df[depth].min(), df[depth].max())
    else:
        ax.set_ylim(ylim[0], ylim[1])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(labelsize=13)
    ax.set_title(title, fontsize=16, fontweight='bold')
    if fill_log:
        # Get x axis range.
        left_value, right_value = ax.get_xlim()
        span = abs(left_value - right_value)
        # Get log value.
        curve = df[log]
        # Assign color map.
        cmap = plt.get_cmap(cmap)
        # Create array of values to divide up the area under curve.
        color_index = np.arange(left_value, right_value, span / 100)
        # Loop through each value in the color_index.
        for index in sorted(color_index):
            index_value = (index - left_value) / span
            color = cmap(index_value)  # Obtain color for color index value.
            plt.fill_betweenx(df[depth], 0, curve, where=curve >= index, color=color)
    if show:
        plt.show()


def rock_physics(df=None, vp_col=None, vs_col=None, den_col=None,
                 switch=None):
    """
    Compute rock-physics parameters.
    :param df: (Pandas.DataFrame) - Well log data frame.
    :param vp_col: (String) - P-wave velocity column name.
    :param vs_col: (String) - S-wave velocity column name.
    :param den_col: (String) - Density column name.
    :param switch: (List of bools) - Default is [True, True, True], which means to compute P-wave impedance, S-wave
                   impedance and Shear Modulus, Bulk Modulus, Young's Modulus and Poisson's Ratio.
    :return: df: (Pandas.DataFrame) - Well log data frame with rock-physics parameters.
    """
    if switch is None:
        switch = [True, True, True]
    if switch[0]:
        df['Ip'] = df[vp_col] * df[den_col]  # P-wave impedance.
    if switch[1]:
        df['Is'] = df[vs_col] * df[den_col]  # S-wave impedance.
    if switch[2]:
        df['Shear Modulus'] = df[vs_col]**2 * df[den_col]  # Shear modulus.
        lame = df[vp_col] ** 2 * df[den_col] - 2 * df['Shear Modulus']  # Lame constant.
        df['Bulk Modulus'] = (3 * lame + 2 * df['Shear Modulus']) / 3  # Bulk modulus.
        df["Poisson's Ratio"] = lame / (2 * (lame * df['Shear Modulus']))  # Poisson's ratio.
        df["Young's Modulus"] = 2 * df['Shear Modulus'] * (1 + df["Poisson's Ratio"])  # Young's modulus.
    return df


def cube2well(cube_file=None, cube_name=None, header_x=73, header_y=77, scl_x=1, scl_y=1,
              df=None, x_col=None, y_col=None, z_col=None, well_coord=None,
              w_x=25.0, w_y=25.0, w_z=2.0):
    """
    Get up-hole trace data from cube and add them to well log data frame.
    :param cube_file: (String) - Cube file name. SEG-Y format.
    :param cube_name: (String) - Cube data name, which will be a new column name in well log data frame.
    :param header_x: (Integer) - Default is 73. Trace x coordinate's byte position in trace header.
                     73: source X, 181: X cdp.
    :param header_y: (Integer) - Default is 77. Trace y coordinate's byte position in trace header.
                     77: source Y, 185: Y cdp.
    :param scl_x: (Float) - The trace x coordinates will multiply this parameter.
                  Default is 1, which means not to scale the trace x coordinates read from trace header.
                  For example, if scl_x=0.1, the trace x coordinates from trace header will multiply 0.1.
    :param scl_y: (Float) - The trace y coordinates will multiply this parameter.
                  Default is 1, which means no to scale the trace y coordinates read from trace header.
                  For example, if scl_y=0.1, the trace y coordinates from trace header will multiply 0.1.
    :param df: (Pandas.Dataframe) - Time domain well log data frame, which has to contain x, y and z coordinate column,
                                    or at least z coordinate column but requires manually input well_coord.
    :param x_col: (String) - X-coordinate column name.
    :param y_col: (String) - Y-coordinate column name.
    :param z_col: (String) - Z-coordinate column name.
    :param well_coord: (List of floats) - Well location coordinates in the form of [well_x, well_y] (e.g. [10.0, 50.0]).
                       If the well log data frame does not have x and y columns, then all well log samples' coordinates
                       will be well_coord.
                       This parameter has no effect when x_col or y_col is not None.
    :param w_x: (Float) - Default is 25.0. The window's x size in which the well xy coordinates and cube xy coordinates
                will be matched.
    :param w_y: (Float) - Default is 25.0. The window's y size in which the well xy coordinates and cube xy coordinates
                will be matched.
    :param w_z: (Float) - Default is 2.0. The window's z size in which the well z coordinates and cube z coordinates
                will be matched.
    :return: df: (Pandas.Dataframe) - Well log data frame with a new column of up-hole trace data from cube.
    """
    # Load cube.
    with segyio.open(cube_file) as f:
        f.mmap()  # Memory map file for faster reading.
        cube = segyio.tools.cube(f)  # Load cube data.
        x = np.zeros(shape=(f.tracecount,), dtype='float32')  # Initiate trace x-coordinates.
        y = np.zeros(shape=(f.tracecount,), dtype='float32')  # Initiate trace y-coordinates.
        for i in range(f.tracecount):
            x[i] = f.header[i][header_x] * scl_x  # Get x-coordinates from trace header.
            y[i] = f.header[i][header_y] * scl_y  # Get y-coordinates from trace header.
        x = x.reshape([len(f.ilines), len(f.xlines)], order='C')  # Re-shape x-coordinates array to match the cube.
        y = y.reshape([len(f.ilines), len(f.xlines)], order='C')  # Re-shape y-coordinates array to match the cube.
        t = f.samples  # Get sampling time.
        print('Cube info:')
        print('X range: %.2f-%.2f [%d samples]' % (np.amin(x), np.amax(x), len(f.ilines)))
        print('Y range: %.2f-%.2f [%d samples]' % (np.amin(y), np.amax(y), len(f.xlines)))
        print('Z range: %.1fms-%.1fms [%d samples]' % (np.amin(t), np.amax(t), len(t)))
    f.close()
    # Get well log's 3D coordinates.
    if x_col is None and y_col is None and well_coord is not None:
        well_x = well_coord[0] * np.ones(len(df))
        well_y = well_coord[1] * np.ones(len(df))
    else:
        well_x = df[x_col].values
        well_y = df[y_col].values
    well_z = df[z_col].values
    print('Well info:')
    print('X range: %.2f-%.2f' % (np.amin(well_x), np.amax(well_x)))
    print('Y range: %.2f-%.2f' % (np.amin(well_y), np.amax(well_y)))
    print('Z range: %.1fms-%.1fms [%d samples]' % (np.amin(well_z), np.amax(well_z), len(well_z)))
    # Match well log coordinates with cube data coordinates.
    dist_xy = scipy.spatial.distance.cdist(np.c_[well_x, well_y],
                                           np.c_[x.ravel(order='C'), y.ravel(order='C')],
                                           metric='euclidean')  # xy plane distance map.
    dist_z = scipy.spatial.distance.cdist(np.reshape(well_z, (-1, 1)),
                                          np.reshape(t, (-1, 1)),
                                          metric='minkowski', p=1)  # z-direction distance.
    indx, indy = np.unravel_index(np.argmin(dist_xy, axis=1), x.shape, order='C')
    indz = np.argmin(dist_z, axis=1)
    dist_xy_min = np.amin(dist_xy, axis=1)
    dist_z_min = np.amin(dist_z, axis=1)
    ixy = np.squeeze(np.argwhere(dist_xy_min < math.sqrt(w_x ** 2 + w_y ** 2)))
    iz = np.squeeze(np.argwhere(dist_z_min < w_z))
    ind = np.intersect1d(ixy, iz)
    if len(ind) == 0:
        raise ValueError('The well log has no sample can match the cube, please check the coordinates.')
    # Get data from the cube.
    df.loc[ind, cube_name] = cube[indx[ind], indy[ind], indz[ind]]
    return df


def outlier_filter(df=None, condition=None, delete_inf=False, delete_none=False, delete_nan=False, remove_row=False):
    """
    Replace outliers with missing value (NaN), or remove rows with outliers in well log data frame.
    :param df: (Pandas.Dataframe) - Well log data frame.
    :param condition: (Dictionary) - Filter condition with well log column names and filter's ends.
                      For example, condition={'POR': [5, 60], 'RT': [0, None], 'SW': [None, 100]} means a band-pass
                      filter that keeps porosity values from 5% to 60%, a high-pass filter that keeps resistivity values
                      no less than 0 ohm.m, and a low-pass filter that keeps water saturation values no more than 100%.
    :param delete_inf: (Bool) - Default is False. Whether to replace (+-)INF with NaN, or remove the whole row with
                       (+-)INF (control by remove_row).
    :param delete_none: (Bool) - Default is False. Whether to replace None with NaN. Useful when importing data frame
                        from SQL database.
    :param delete_nan: (Bool) - Default is False. Whether to remove rows with any NaN.
    :param remove_row: (Bool) - Default is False, whether to remove the whole row with any outlier.
    :return: df: (Pandas.Dataframe) - Filtered well log data frame.
    """
    # Replace INF with NaN or remove rows with INF.
    if delete_inf:
        if remove_row:  # Remove rows with INF or -INF.
            ind = [x for x in range(len(df)) if (df.iloc[x, :] == np.inf).any() or (df.iloc[x, :] == -np.inf).any()]
            df.drop(index=ind, inplace=True)
            df.reset_index(drop=True, inplace=True)
        else:  # Replace INF or -INF with NaN.
            df.replace([-np.inf, np.inf], np.nan, inplace=True)
    # Replace None with NaN.
    if delete_none:
        df.fillna(np.nan, inplace=True)
    # Remove rows with any NaN.
    if delete_nan:
        df.dropna(axis='index', how='any', inplace=True)
        df.reset_index(drop=True, inplace=True)
    # Filter outliers, replace outliers with NaN.
    if condition is not None:
        col = list(condition.keys())  # Get column names from condition dictionary.
        filters = list(condition.values())  # Get filter values from condition dictionary.
        for i in range(len(condition)):
            l, r = filters[i][0], filters[i][1]  # Left and right end of the filter.
            if l is not None:
                ind = [x for x in range(len(df)) if df.loc[x, col[i]] < l]
                if remove_row:
                    df.drop(index=ind, inplace=True)
                    df.reset_index(drop=True, inplace=True)
                else:
                    df.loc[ind, col[i]] = np.nan
            if r is not None:
                ind = [x for x in range(len(df)) if df.loc[x, col[i]] > r]
                if remove_row:
                    df.drop(index=ind, inplace=True)
                    df.reset_index(drop=True, inplace=True)
                else:
                    df.loc[ind, col[i]] = np.nan
    return df


def check_info(df=None, log_col='all'):
    """
    Visualize well log info of [Log name, Samples amount, Missing value amount, Min, Max, Mean, Standard deviation].
    :param df: (Pandas.Dataframe) - Well log data frame.
    :param log_col: (String or list of strings) - Default is 'all', which means taking all except the first column to
                    visualize info. Also you can manually enter the log names as a list, or just a string for one log.
    """
    # Use pretty table to visualize info.
    table = PrettyTable()
    # Set column names of the table.
    table.field_names = ['Log name', 'Samples', 'Missing', 'Min', 'Max', 'Mean', 'Std']
    table.float_format = '.2'
    # When log_col is a string...
    if isinstance(log_col, str):
        if log_col == 'all':
            log_col = list(df.columns)[1:]
        else:
            log_col = [log_col]
    # Add rows to the table.
    for i in range(len(log_col)):
        log_name = log_col[i]
        n_sample = len(df[log_name])
        n_miss = df[log_name].isna().sum()
        minimum = df[log_name].min()
        maximum = df[log_name].max()
        mean = df[log_name].mean()
        std = df[log_name].std()
        table.add_row([log_name, n_sample, n_miss, minimum, maximum, mean, std])
    # Display the table.
    print(table)

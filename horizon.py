import os
import sys
import time
import math
import segyio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance
from scipy import interpolate
from sklearn import preprocessing


def horizon_interp(df=None, x_step=25.0, y_step=25.0, method='linear', visualize=True,
                   x_col=None, y_col=None, t_col=None,
                   infer=True, xy_range=None, xy_dtype='float32'):
    """
    Interpolate horizon on a regular grid.
    :param df: (pandas.DataFrame) - Horizon data frame which contains ['inline', 'xline', 'x', 'y', 't'] columns.
    :param x_step: (Float) - Default is 25.0. x coordinate step of the regular grid.
    :param y_step: (Float) - Default is 25.0. y coordinate step of the regular grid.
    :param method: (String) - Default is 'linear'. Method of interpolation. One of {'linear', 'nearest', 'cubic'}.
    :param visualize: (Bool) - Default is True. Whether to visualize the interpolation result.
    :param x_col: (String) - x-coordinate column name.
    :param y_col: (String) - y-coordinate column name.
    :param t_col: (String) - t-coordinate column name.
    :param infer: (Bool) - Default is True. Whether to infer x and y range from input data frame.
    :param xy_range: (List of floats) - When infer is False, require manually input x and y range
                     [x_min, x_max, y_min, y_max].
    :param xy_dtype: (String) - Default is 'float32'. Data types of output x and y data.
    :return df_new: (pandas.DataFrame) - Interpolated horizon data frame.
    """
    # Get min and max of x and y.
    print('Interpolating horizon...')
    if infer:
        xmin, xmax = np.nanmin(df[x_col].values), np.nanmax(df[x_col].values)
        ymin, ymax = np.nanmin(df[y_col].values), np.nanmax(df[y_col].values)
        print('Inferred x range: %.2f-%.2f' % (xmin.item(), xmax.item()))
        print('Inferred y range: %.2f-%.2f' % (ymin.item(), ymax.item()))
    else:
        xmin, xmax = xy_range[0], xy_range[1]
        ymin, ymax = xy_range[2], xy_range[3]
        print('Defined x range: %.2f-%.2f' % (xmin, xmax))
        print('Defined y range: %.2f-%.2f' % (ymin, ymax))
    # Get 2D coordinates of control points.
    df_coord = df[[x_col, y_col, t_col]].copy()
    df_coord.dropna(axis='index', how='any', inplace=True)
    xy = df_coord[[x_col, y_col]].values
    t = df_coord[t_col].values
    # Create new 2D coordinates to interpolate.
    xnew = np.linspace(xmin, xmax, int((xmax - xmin) / x_step) + 1, dtype='float32')
    ynew = np.linspace(ymin, ymax, int((ymax - ymin) / y_step) + 1, dtype='float32')
    xnew, ynew = np.meshgrid(xnew, ynew, indexing='ij')
    # Interpolate.
    tnew = interpolate.griddata(points=xy, values=t, xi=(xnew, ynew), method=method)
    # Output interpolated horizon data.
    d = {x_col: xnew.ravel(order='C'),
         y_col: ynew.ravel(order='C'),
         t_col: tnew.ravel(order='C')}
    df_new = pd.DataFrame(d, dtype='float32')
    nan_present = df_new.isna().any().any()
    if nan_present:
        df_temp = df_new.dropna(axis='index', how='any')
        tnew = interpolate.griddata(points=df_temp[[x_col, y_col]].values, values=df_temp[t_col].values,
                                    xi=(xnew, ynew), method='nearest')
        df_new[t_col] = tnew.ravel(order='C')
    nan_present = df_new.isna().any().any()
    if nan_present:
        print('Failed. Cannot eliminate NaN.')
    else:
        print('Done.')
    df_new[[x_col, y_col]] = df_new[[x_col, y_col]].astype(xy_dtype)
    if visualize:
        # Visualize.
        plt.figure()
        plt.title('Interpolation Result')
        cset = plt.contourf(xnew, ynew, tnew, 8, cmap='rainbow')
        plt.colorbar(cset)
        plt.show()
    return df_new


def visualize_horizon(df=None, x_name=None, y_name=None, value_name=None, deltax=25.0, deltay=25.0, cmap='seismic_r',
                      vmin=None, vmax=None, nominal=False, class_code=None, class_label=None, fig_name=None,
                      cbar_label=None, axe_aspect='equal', show=True):
    """
    Visualize horizon data.
    :param df: (pandas.DataFrame) - Horizon data frame which contains ['x', 'y', 'value1', 'value2', '...'] columns.
    :param x_name: (String) - x-coordinate column name.
    :param y_name: (String) - y-coordinate column name.
    :param value_name: (String) - Column name of the values over which the plot is drawn.
    :param deltax: (Float) - Default is 25.0. x-coordinate interval of the regular grid.
    :param deltay: (Float) - Default is 25.0. y-coordinate interval of the regular grid.
    :param cmap: (String) - Default is 'seismic_r'. The color map.
    :param vmin: (Float) - Minimum value of the continuous color bar.
    :param vmax: (Float) - Maximum value of the continuous color bar.
    :param nominal: (Bool) - Default is False. Whether the data is nominal (discrete).
    :param class_code: (List of integers) - The class codes. (e.g. [0, 1, 2])
    :param class_label: (List of strings) - The class labels. (e.g. ['Mudstone', 'Shale', 'Sandstone']).
    :param fig_name: (String) - Default is 'Result'. Figure name.
    :param cbar_label: (String) - The color bar label. Default is to use value_name as color bar label.
    :param axe_aspect: (String) - Default is 'equal'. The aspect ratio of the axes.
                       'equal': ensures an aspect ratio of 1. Pixels will be square (unless pixel sizes are explicitly
                       made non-square in data coordinates using extent).
                       'auto': The axes is kept fixed and the aspect is adjusted so that the data fit in the axes. In
                       general, this will result in non-square pixels.
    :param show: (Bool) - Default is True. Whether to show plot.
    """
    # Construct regular grid to show horizon.
    x = df[x_name].values
    xmin, xmax = np.amin(x), np.amax(x)
    y = df[y_name].values
    ymin, ymax = np.amin(y), np.amax(y)
    xnew = np.linspace(xmin, xmax, int((xmax - xmin) / deltax) + 1)
    ynew = np.linspace(ymin, ymax, int((ymax - ymin) / deltay) + 1)
    xgrid, ygrid = np.meshgrid(xnew, ynew)
    xgrid = xgrid.ravel(order='F')
    ygrid = ygrid.ravel(order='F')
    df_grid = pd.DataFrame({x_name: xgrid, y_name: ygrid})
    # Merge irregular horizon on regular grid.
    df_horizon = pd.merge(left=df_grid, right=df, how='left', on=[x_name, y_name])
    data = df_horizon[value_name].values
    x = df_horizon[x_name].values
    y = df_horizon[y_name].values
    # Plot horizon.
    plt.figure(figsize=(12, 8))
    if fig_name is None:
        plt.title('Result', fontsize=20)
    else:
        plt.title(fig_name, fontsize=22)
    plt.ticklabel_format(style='plain')
    extent = [np.amin(x), np.amax(x), np.amin(y), np.amax(y)]
    plt.imshow(data.reshape([len(ynew), len(xnew)], order='F'),
               origin='lower', aspect=axe_aspect, vmin=vmin, vmax=vmax,
               cmap=plt.cm.get_cmap(cmap), extent=extent)
    plt.xlabel(x_name, fontsize=18)
    plt.ylabel(y_name, fontsize=18)
    plt.tick_params(labelsize=14)
    if cbar_label is None:
        cbar_label = value_name
    if nominal:
        class_code.sort()
        tick_min = (max(class_code) - min(class_code)) / (2 * len(class_code)) + min(class_code)
        tick_max = max(class_code) - (max(class_code) - min(class_code)) / (2 * len(class_code))
        tick_step = (max(class_code) - min(class_code)) / len(class_code)
        ticks = np.arange(start=tick_min, stop=tick_max + tick_step, step=tick_step)
        cbar = plt.colorbar(ticks=ticks)
        cbar.set_label(cbar_label, fontsize=18)
        cbar.ax.set_yticklabels(class_label)
        cbar.ax.tick_params(axis='y', labelsize=16)
    else:
        cbar = plt.colorbar()
        cbar.ax.tick_params(axis='y', labelsize=16)
        cbar.set_label(cbar_label, fontsize=18)
    if show:
        plt.show()


def horizon_log(df_horizon=None, df_well_coord=None, log_file_path=None, sep='\t', well_name_col=None,
                horizon_x_col=None, horizon_y_col=None, horizon_z_col=None,
                log_x_col=None, log_y_col=None, log_z_col=None, log_value_col=None, log_abnormal_value=None,
                log_file_suffix='.txt', print_progress=False, w_x=25.0, w_y=25.0, w_z=2.0):
    """
    Mark log values on horizon. Only for vertical well now.
    :param df_horizon: (pandas.DataFrame) - Horizon data frame which contains ['x', 'y', 'z'] columns.
    :param df_well_coord: (pandas.DataFrame) - Default is None, which means not to use a well coordinates data frame and
                          the log files must contain well x and y columns. If not None, then this is a well coordinates
                          data frame which contains ['well_name', 'well_x', 'well_y'] columns.
    :param log_file_path: (String) - Time domain well log file directory.
    :param sep: (String) - Default is '\t'. Column delimiter in log files.
    :param well_name_col: (String) - Well name column name in df_well_coord.
    :param horizon_x_col: (String) - Horizon x-coordinate column name in df_horizon.
    :param horizon_y_col: (String) - Horizon y-coordinate column name in df_horizon.
    :param horizon_z_col: (String) - Horizon z-coordinate column name in df_horizon.
    :param log_x_col: (String) - Well x-coordinate column name.
    :param log_y_col: (String) - Well y-coordinate column name.
    :param log_z_col: (String) - Well log two-way time column name.
    :param log_value_col: (String) - Well log value column name.
    :param log_abnormal_value: (Float) - Abnormal value in log value column.
    :param log_file_suffix: (String) - Default is '.txt'. Well log file suffix.
    :param print_progress: (Bool) - Default is False. Whether to print progress.
    :param w_x:  (Float) - Default is 25.0.
                 Size of x window in which the well xy-coordinates and horizon xy-coordinates will be matched.
    :param w_y: (Float) - Default is 25.0.
                Size of y window in which the well xy-coordinates and horizon xy-coordinates will be matched.
    :param w_z: (Float) - Default is 2.0.
                Size of z window in which the well z-coordinates and horizon z-coordinates will be matched.
    :return: df_out: (pandas.DataFrame) - Output data frame which contains ['x', 'y', 'z', 'log value', 'well name']
                     columns.
    """
    # Initialize output data frame as a copy of horizon data frame.
    df_out = df_horizon.copy()
    # List of well log files.
    log_file_list = os.listdir(log_file_path)
    # Mark log values on horizon.
    for log_file in log_file_list:
        # Load well log data.
        df_log = pd.read_csv(os.path.join(log_file_path, log_file), delimiter=sep)
        if log_abnormal_value is not None:
            drop_ind = [x for x in range(len(df_log)) if df_log.loc[x, log_value_col] == log_abnormal_value]
            df_log.drop(index=drop_ind, inplace=True)
            df_log.reset_index(drop=True, inplace=True)
        # Get well name.
        well_name = log_file[:-len(log_file_suffix)]
        # Print progress.
        if print_progress:
            sys.stdout.write('\rMatching well %s' % well_name)
        if df_well_coord is not None:
            # Get well coordinates from well coordinate file.
            [well_x, well_y] = \
                np.squeeze(df_well_coord[df_well_coord[well_name_col] == well_name][[log_x_col, log_y_col]].values)
        else:
            well_x, well_y = df_log.loc[0, log_x_col], df_log.loc[0, log_y_col]
        # Get horizon coordinates.
        horizon_x = df_horizon[horizon_x_col].values
        horizon_y = df_horizon[horizon_y_col].values
        # Compute distance map between well and horizon.
        xy_dist = np.sqrt((horizon_x - well_x) ** 2 + (horizon_y - well_y) ** 2)
        # Get array index of minimum distance in distance map. This is the horizon coordinate closest to the well.
        idx_xy = np.argmin(xy_dist)
        if xy_dist[idx_xy] < math.sqrt(w_x ** 2 + w_y ** 2):
            # Get horizon two-way time at the closest point to the well.
            horizon_t = df_horizon.loc[idx_xy, horizon_z_col]
            # Get well log two-way time.
            log_t = df_log[log_z_col].values
            # Compute distances between well log two-way time and horizon two-way time at the closest point to the well.
            t_dist = np.abs(log_t - horizon_t)
            # Get array index of the minimum distance. This the vertically closest point of the well log to the horizon.
            idx_t = np.argmin(t_dist)
            if t_dist[idx_t] < w_z:
                # Get log value.
                log_value = df_log.loc[idx_t, log_value_col]
                # Mark log value on horizon.
                df_out.loc[idx_xy, log_value_col] = log_value
                # Mark well name.
                df_out.loc[idx_xy, 'WellName'] = well_name
    # Drop NaN.
    df_out.dropna(axis='index', how='any', subset=[log_value_col], inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    return df_out


def plot_markers(df=None, x_col=None, y_col=None, class_col=None, wellname_col=None,
                 class_code=None, class_label=None, colors=None,
                 annotate=True, anno_color='k', anno_fontsize=12, anno_shift=None, show=False):
    """
    Plot markers on horizon. For nominal markers only.
    :param df: (pandas.Dataframe) - Marker data frame which contains ['x', 'y', 'class', 'well name'] columns.
    :param x_col: (String) - x-coordinate column name.
    :param y_col: (String) - y-coordinate column name.
    :param class_col: (String) - Class column name.
    :param wellname_col: (String) - Well name column name.
    :param class_code: (List of integers) - Class codes of the markers.
    :param class_label: (List of strings) - Label names of the markers.
    :param colors: (List of strings) - Color names of the markers.
    :param annotate: (Bool) - Default is True. Whether to annotate well names beside well markers.
    :param anno_color: (String) - Annotation text color.
    :param anno_fontsize: (Integer) - Default is 12. Annotation text font size.
    :param anno_shift: (List of floats) - Default is [0, 0]. Annotation text coordinates.
    :param show: (Bool) - Default is False. Whether to show markers.
    """
    # Get class codes.
    marker_class = df[class_col].copy()
    marker_class.drop_duplicates(inplace=True)
    marker_class = marker_class.values
    marker_class = marker_class.astype('int')
    marker_class.sort()
    # Plot markers.
    for i in range(len(marker_class)):
        idx = [x for x in range(len(df)) if df.loc[x, class_col] == marker_class[i]]
        sub_frame = df.loc[idx, [x_col, y_col]]
        x, y = sub_frame.values[:, 0], sub_frame.values[:, 1]
        idx = np.squeeze(np.argwhere(class_code == marker_class[i]))
        plt.scatter(x, y, c=colors[idx], edgecolors='k', label=class_label[idx],
                    s=50)
    if annotate:
        # Annotate well names.
        for i in range(len(df)):
            well_name = df.loc[i, wellname_col]
            x_anno = df.loc[i, x_col]
            y_anno = df.loc[i, y_col]
            if anno_shift is None:
                anno_shift = [0, 0]
            plt.annotate(text=well_name, xy=(x_anno, y_anno), xytext=(x_anno - anno_shift[0], y_anno + anno_shift[1]),
                         color=anno_color, fontsize=anno_fontsize)
    plt.legend(loc='upper right', fontsize=15)
    if show:
        plt.show()


def FSDI_horizon(df_horizon=None, df_control=None, coord_col=None, feature_col=None, log_col=None,
                 scale=True, weight=None):
    """
    Feature and Space Distance based Interpolation for horizons.
    :param df_horizon: (pandas.DataFrame) - Horizon data frame, these are points to interpolate, which should
                       contains coordinates columns and features columns.
    :param df_control: (pandas.DataFrame) - Data frame of control points on the horizon,
                                            which should contains coordinates columns and log columns.
    :param coord_col: (List of Strings) - Coordinates columns names. (e.g. ['x', 'y', 't']).
    :param feature_col: (String or list of strings) - Features columns names.
    :param log_col: (String) - Well log column name.
    :param scale: (Bool) - Default is True. Whether to scale coordinates and features.
    :param weight: (List of floats) - Default is that all features (including spatial coordinates) have equal weight.
                                      Weight of spatial coordinates and features, e.g. [1, 1, 1, 2, 2] for
                                      ['x', 'y', 'z', 'amplitude', 'vp'].
    :return: df_horizon: (pandas.DataFrame) - Interpolation result.
             df_horizon_log: (pandas.DataFrame) - Control points.
    """
    # Select columns of horizon data frame.
    if isinstance(feature_col, str):
        feature_col = [feature_col]
    df_h = df_horizon[coord_col + feature_col].copy()
    # Select columns of horizon log data frame.
    if isinstance(log_col, list) and len(log_col) == 1:
        log_col = log_col[0]
    df_l = df_control[coord_col + [log_col]].copy()
    # Match features to control points by coordinates.
    df_l = pd.merge(left=df_l, right=df_h, on=coord_col, how='inner')
    # Assign target columns.
    column = coord_col + feature_col
    # Whether to scale coordinates and features.
    if scale:
        # MinMaxScalar.
        min_max_scalar = preprocessing.MinMaxScaler()
        # Transform (Scale).
        df_h[column] = min_max_scalar.fit_transform(df_h[column].values)  # Points to interpolate.
        df_l[column] = min_max_scalar.transform(df_l[column].values)  # Control points.
    # Get control points' coordinates and features.
    control = df_l[column].values
    # Get control points' log values.
    control_log = df_l[log_col].values
    # Get horizon's coordinates and features.
    horizon = df_h[column].values
    # Compute feature & space distance map.
    if weight is None:
        weight = np.ones(len(column))
    else:
        weight = np.array(weight)
    dist_map = scipy.spatial.distance.cdist(horizon, control, metric='minkowski', p=2, w=weight)
    # Get column index of minimum distance.
    min_idx = np.argmin(dist_map, axis=1)
    # Interpolate log values according to minimum distance.
    df_horizon[log_col] = control_log[min_idx]
    return df_horizon, df_control


def FSDI_interhorizon(feature_file=None, feature_name=None, header_x=73, header_y=77, scl_x=1, scl_y=1,
                      horizon_file=None, horizon_col=None, horizon_x_col=None, horizon_y_col=None, horizon_z_col=None,
                      horizon_dx=25.0, horizon_dy=25.0, horizon_xy_infer=False, horizon_xy_range=None, horizon_sep='\t',
                      log_dir=None, log_x_col=None, log_y_col=None, log_z_col=None, log_value_col=None, log_sep='\t',
                      df_well_loc=None, well_name_col=None, log_file_suffix='.txt', log_abnormal=None,
                      dp=None, weight=None, fill_value=None, init_value=None, tight_frame=True,
                      output_file=None, output_file_suffix='.dat'):
    """
    Feature and Space Distance based Interpolation for inter-horizon.
    :param feature_file: (String or list of strings) - Feature file. SEG-Y file name with absolute file path.
                         For multiple file, input file name list such as ['.../.../a.sgy', '.../.../b.sgy'].
    :param feature_name: (String or list of strings) - Feature name. Must correspond to seismic files. For multiple
                         features, input name list such as ['a', 'b'].
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
    :param horizon_file: (List of strings) - Horizon file list. Notice that in each horizon file there must be 3 columns
                         of x, y and t data. The horizon data coordinates must match with the seismic data coordinates,
                         or at least in the same range.
    :param horizon_col: (List of strings) - Default is None, which means the horizon files have column names.
                        If the horizon files have no column names. Define column names of each horizon file with this
                        parameter (e.g. ['x', 'y', 't']).
    :param horizon_x_col: (String) - X-coordinate column name of horizon file.
    :param horizon_y_col: (String) - Y-coordinate column name of horizon file.
    :param horizon_z_col: (String) - Z-coordinate column name of horizon file.
    :param horizon_dx: (Float or integer) - Default is 25.0. X-coordinate spacing of every horizon.
    :param horizon_dy: (Float or integer) - Default is 25.0. Y-coordinate spacing of every horizon.
    :param horizon_xy_infer: (Bool) - Default is True. Whether to infer x and y range from horizon data.
    :param horizon_xy_range: (List of floats) - When infer is False, require manually input x and y range
                             [x_min, x_max, y_min, y_max].
    :param horizon_sep: (String) - Default is '\t'. Horizon file column delimiter.
    :param log_dir: (String) - Time domain well log file directory.
    :param log_x_col: (String) - Well log x-coordinate column name.
    :param log_y_col: (String) - Well log y-coordinate column name.
    :param log_z_col: (String) - Well log z-coordinate column name.
    :param log_value_col: (String) - Well log value column name.
    :param log_sep: (String) - Default is '\t'. Well log file column delimiter.
    :param df_well_loc: (Pandas.DataFrame) - Data frame with well names and well location coordinates.
    :param well_name_col: (String) - Default is 'well_name'. Well name column name in well location file.
    :param log_file_suffix: (String) - Default is '.txt'. Well log file suffix.
    :param log_abnormal: (Float) - Abnormal value in well log value column.
    :param dp: (Float) - Default is None, which is to compute a self-adaptive pseudo-inner-horizon percentage step from
                         input horizon data. Can also be defined manually such as 0.02, which means the percentage step
                         is 2%.
    :param weight: (List of floats) - Default is that all features (including spatial coordinates) have equal weight.
                                      Weight of spatial coordinates and features, e.g. [1, 1, 1, 2, 2] for
                                      ['x', 'y', 'z', 'amplitude', 'vp'].
    :param fill_value: (Float or integer) - Fill value outside all horizons.
    :param init_value: (Float or integer) - Initial value inside horizons.
    :param tight_frame: (Bool) - Default is True, which is to cut a tight box as an envelope of interpolated inter-
                        horizon data.
    :param output_file: (String) - Output file name with absolute file path.
    :param output_file_suffix: (String) - Default is '.dat'. The suffix of the output file.
    :return: cube_itp: (numpy.3darray) - Interpolation result.
    """
    t1 = time.perf_counter()  # Timer.
    # Read feature file.
    if isinstance(feature_file, str):
        feature_file = [feature_file]
    if isinstance(feature_file, str) is False and isinstance(feature_file, list) is False:
        raise ValueError('Feature file must be string or list of strings.')
    if isinstance(feature_name, str):
        if len(feature_file) != 1:
            raise ValueError('The number of feature names must match the number of feature files.')
    if isinstance(feature_name, list):
        if len(feature_name) != len(feature_file):
            raise ValueError('The number of feature names must match the number of feature files.')
    if isinstance(feature_name, str) is False and isinstance(feature_name, list) is False:
        raise ValueError('Feature name must be string or list of strings.')
    feature = []  # Initiate list to store feature data.
    for file in feature_file:
        with segyio.open(file) as f:
            print('Read feature data from file: ', file)
            # Memory map file for faster reading (especially if file is big...)
            f.mmap()
            # Print file information.
            print('File info:')
            print('inline range: %d-%d [%d lines]' % (f.ilines[0], f.ilines[-1], len(f.ilines)))
            print('crossline range: %d-%d [%d lines]' % (f.xlines[0], f.xlines[-1], len(f.xlines)))
            print('Depth range: %dms-%dms [%d samples]' % (f.samples[0], f.samples[-1], len(f.samples)))
            dt = segyio.tools.dt(f) / 1000
            print('Sampling interval: %.1fms' % dt)
            print('Total traces: %d' % f.tracecount)
            # Read seismic data.
            cube = segyio.tools.cube(f)
            # When reading the last file, extract sampling time and coordinates.
            if file == feature_file[-1]:
                # Read sampling time.
                t = f.samples
                # Extract trace coordinates from trace header.
                x = np.zeros(shape=(f.tracecount,), dtype='float32')
                y = np.zeros(shape=(f.tracecount,), dtype='float32')
                for i in range(f.tracecount):
                    sys.stdout.write('\rExtracting trace coordinates: %.2f%%' % ((i + 1) / f.tracecount * 100))
                    x[i] = f.header[i][header_x] * scl_x
                    y[i] = f.header[i][header_y] * scl_y
                sys.stdout.write('\n')
                # Re-shape the trace coordinates array to match the feature data cube.
                x_2d = x.reshape([len(f.ilines), len(f.xlines)], order='C')
                y_2d = y.reshape([len(f.ilines), len(f.xlines)], order='C')
                # The corresponding coordinates index.
                x_ind = np.linspace(0, len(f.ilines)-1, len(f.ilines), dtype='int32')
                y_ind = np.linspace(0, len(f.xlines)-1, len(f.xlines), dtype='int32')
                x_ind, y_ind = np.meshgrid(x_ind, y_ind, indexing='ij')
                x_ind = np.ravel(x_ind, order='C')
                y_ind = np.ravel(y_ind, order='C')
        f.close()
        feature.append(cube)
    # Read horizon files.
    if isinstance(horizon_file, list) is False:
        raise ValueError('Horizon file must be list of strings.')
    if isinstance(horizon_file, list) and len(horizon_file) == 1:
        raise ValueError('The number of horizons must >=2. If you want to process single horizon, '
                         'use function FSDI_horizon.')
    horizon = []  # Initiate list to store horizon data.
    for file in horizon_file:
        sys.stdout.write('Reading horizon file %s...' % file)
        if horizon_col is not None:
            df_horizon = pd.read_csv(file, names=horizon_col, delimiter=horizon_sep)
        else:
            df_horizon = pd.read_csv(file, delimiter=horizon_sep)
        sys.stdout.write(' Done.\n')
        sys.stdout.write('Checking information...\n')
        sys.stdout.write('X and Y coordinate match seismic data:')
        if np.equal(df_horizon[horizon_x_col].values, x).all() and \
                np.equal(df_horizon[horizon_y_col].values, y).all():
            sys.stdout.write(' True.\n')
            sys.stdout.write('Z coordinate NaN present:')
            if df_horizon[horizon_z_col].isna().any():
                sys.stdout.write(' True. %d NaN detected.\n' % df_horizon[horizon_z_col].isna().sum())
                df_horizon = horizon_interp(df_horizon, x_col=horizon_x_col, y_col=horizon_y_col, t_col=horizon_z_col,
                                            x_step=horizon_dx, y_step=horizon_dy, visualize=False)
            else:
                sys.stdout.write(' False.\n')
        else:
            sys.stdout.write(' False.\n')
            if horizon_xy_infer is False and horizon_xy_range is not None:
                sys.stdout.write('Trying the input horizon xy range.\n')
                df_horizon = horizon_interp(df_horizon, x_col=horizon_x_col, y_col=horizon_y_col, t_col=horizon_z_col,
                                            x_step=horizon_dx, y_step=horizon_dy, visualize=False,
                                            infer=horizon_xy_infer, xy_range=horizon_xy_range)
                if np.equal(df_horizon[horizon_x_col].values, x).all() and \
                        np.equal(df_horizon[horizon_y_col].values, y).all():
                    raise ValueError("Horizon xy coordinates can't match the cube with manually input xy range.")
            else:
                raise ValueError("Horizon xy coordinates can't match the cube coordinates, please check.")
        horizon.append(df_horizon)
    # Generate pseudo-inner-horizons.
    horizon_new = []
    z_ind_list = []
    z_ind_min = 1e30
    z_ind_max = -999
    for i in range(len(horizon) - 1):
        # Compute thickness between horizons.
        th = horizon[i+1][horizon_z_col].values - horizon[i][horizon_z_col].values
        # Get the maximum thickness, notice that NaN may present (use np.nanmax instead of np.amax).
        th_max = np.nanmax(th)
        if dp is None:
            per = np.linspace(0, 1, num=int(round(1/(dt/th_max))), dtype='float32')
        else:
            per = np.linspace(0, 1, num=int(round(1/dp)), dtype='float32')
        for j in range(len(per)):
            sys.stdout.write('\rGenerating pseudo-inner-horizons[series %d/%d]: %.2f%% [%d/%d]' %
                             (i+1, len(horizon)-1, (j+1)/len(per)*100, j+1, len(per)))
            df_temp = horizon[i].copy()
            t_new = horizon[i][horizon_z_col].values + th * per[j]  # The depth of pseudo-inner-horizons.
            df_temp[horizon_z_col] = t_new
            # Add features to pseudo-inner-horizons.
            dist_map = scipy.spatial.distance.cdist(np.reshape(df_temp[horizon_z_col].values, (-1, 1)),
                                                    np.reshape(t, (-1, 1)), metric='minkowski', p=1)
            z_ind = np.argmin(dist_map, axis=1)  # The horizon depth indexes of every trace in feature cubes.
            if np.amin(z_ind) < z_ind_min:
                z_ind_min = np.amin(z_ind)
            if np.amax(z_ind) > z_ind_max:
                z_ind_max = np.amax(z_ind)
            z_ind_list.append(z_ind)
            for k in range(len(feature)):
                if len(feature) == 1:
                    df_temp[feature_name] = feature[k][x_ind, y_ind, z_ind]
                else:
                    df_temp[feature_name[k]] = feature[k][x_ind, y_ind, z_ind]
            horizon_new.append(df_temp)
    sys.stdout.write('\n')
    # Initiate interpolation result array.
    if tight_frame:  # Tight box containing all horizons.
        t = t[z_ind_min: z_ind_max + 1]
        cube_itp = np.ones([feature[0].shape[0], feature[0].shape[1], len(t)], dtype='float32') * fill_value
        if output_file is not None:
            np.savetxt(output_file[:-len(output_file_suffix)] + '_depth.txt', t.reshape(-1, 1))
        for i in range(len(x_ind)):
            sys.stdout.write('\rInitiating array for FSDI interpolation:%.2f%%' % ((i+1) / len(x_ind) * 100))
            cube_itp[x_ind[i], y_ind[i], z_ind_list[0][i]-z_ind_min:z_ind_list[-1][i]+1-z_ind_min] = init_value
        sys.stdout.write('\n')
        print('\tTight frame: ', tight_frame)
        print('\tDepth range: %dms-%dms [%d samples]' % (t[0], t[-1], int((t[-1] - t[0]) / dt + 1)))
    else:
        cube_itp = np.ones(feature[0].shape, dtype='float32') * fill_value
        if output_file is not None:
            np.savetxt(output_file[:-len(output_file_suffix)] + '_depth.txt', t.reshape(-1, 1))
        for i in range(len(x_ind)):
            sys.stdout.write('\rInitiating array for FSDI interpolation:%.2f%%' % ((i+1) / len(x_ind) * 100))
            cube_itp[x_ind[i], y_ind[i], z_ind_list[0][i]:z_ind_list[-1][i]+1] = init_value
        sys.stdout.write('\n')
        print('\tTight frame:', tight_frame)
        print('\tDepth range: %dms-%dms [%d samples]' % (t[0], t[-1], int((t[-1] - t[0]) / dt + 1)))
    # FSDI on pseudo-inner-horizons.
    for i in range(len(horizon_new)):
        sys.stdout.write('\rInterpolating: %.2f%% [%d/%d]' % ((i+1)/len(horizon_new)*100, i+1, len(horizon_new)))
        df_ctp = horizon_log(df_horizon=horizon_new[i], df_well_coord=df_well_loc, log_file_path=log_dir, sep=log_sep,
                             well_name_col=well_name_col, log_x_col=log_x_col, log_y_col=log_y_col,
                             horizon_x_col=horizon_x_col, horizon_y_col=horizon_y_col, horizon_z_col=horizon_z_col,
                             log_z_col=log_z_col, log_value_col=log_value_col, log_file_suffix=log_file_suffix,
                             log_abnormal_value=log_abnormal)
        df_interp, _ = FSDI_horizon(df_horizon=horizon_new[i], df_control=df_ctp,
                                    coord_col=[horizon_x_col, horizon_y_col, horizon_z_col], feature_col=feature_name,
                                    log_col=log_value_col, scale=True, weight=weight)
        if tight_frame:
            cube_itp[x_ind, y_ind, z_ind_list[i] - z_ind_min] = df_interp[log_value_col].values
        else:
            cube_itp[x_ind, y_ind, z_ind_list[i]] = df_interp[log_value_col].values
    sys.stdout.write('\n')
    # Clean abnormal values.
    abnormal_ind = np.argwhere(cube_itp == init_value)
    if len(abnormal_ind):
        sys.stdout.write('%d abnormal value(s) detected. Commencing abnormal value cleaning now...' %
                         (len(abnormal_ind)))
        x_ind, y_ind, z_ind = abnormal_ind[:, 0], abnormal_ind[:, 1], abnormal_ind[:, 2]
        x_ab, y_ab, z_ab = x_2d[x_ind, y_ind], y_2d[x_ind, y_ind], t[z_ind]  # Abnormal values' coordinates.
        if tight_frame:
            x_cube, y_cube, z_cube = np.meshgrid(x_2d[:, 0], y_2d[0, :], t, indexing='ij')
            condition = np.where(np.ravel(cube_itp, order='C') == init_value)
            x_ctp = np.delete(x_cube, condition)
            y_ctp = np.delete(y_cube, condition)
            z_ctp = np.delete(z_cube, condition)
            v_ctp = np.delete(cube_itp, condition)
            clean_value = scipy.interpolate.griddata(points=(x_ctp, y_ctp, z_ctp),
                                                     values=v_ctp,
                                                     xi=(x_ab, y_ab, z_ab), method='nearest')
        else:
            x_cube, y_cube, z_cube_cut = np.meshgrid(x_2d[:, 0], y_2d[0, :], t[z_ind_min: z_ind_max + 1], indexing='ij')
            condition = np.where(np.ravel(cube_itp[:, :, z_ind_min: z_ind_max + 1], order='C') == init_value)
            x_ctp = np.delete(x_cube, condition)
            y_ctp = np.delete(y_cube, condition)
            z_ctp = np.delete(z_cube_cut, condition)
            v_ctp = np.delete(cube_itp[:, :, z_ind_min: z_ind_max + 1], condition)
            clean_value = scipy.interpolate.griddata(points=(x_ctp, y_ctp, z_ctp),
                                                     values=v_ctp,
                                                     xi=(x_ab, y_ab, z_ab), method='nearest')
        cube_itp[x_ind, y_ind, z_ind] = clean_value
        check_abnormal = np.argwhere(cube_itp == init_value)
        if len(check_abnormal):
            sys.stdout.write(' Failed.\n')
            sys.stdout.write('%d abnormal value(s) remained.\n' % len(check_abnormal))
        else:
            sys.stdout.write(' Done.\n')
    # Output interpolation result.
    if output_file is not None:
        sys.stdout.write('Saving interpolation result to file %s...' % output_file)
        cube_out = np.reshape(cube_itp, (cube_itp.shape[0] * cube_itp.shape[1], cube_itp.shape[2]), order='C')
        np.savetxt(output_file, np.c_[x, y, cube_out], delimiter='\t')
        sys.stdout.write(' Done.\n')
    t2 = time.perf_counter()
    # Print process time.
    print('Process time: %.2fs' % (t2 - t1))
    return cube_itp


def cube2horizon(cube_file=None, header_x=73, header_y=77, scl_x=1, scl_y=1,
                 df_horizon=None, hor_x=None, hor_y=None, hor_il=None, hor_xl=None, hor_z=None,
                 value_name=None, match_on='xy', x_win=None, y_win=None, z_win=2.0):
    """
    Get data from a cube to a horizon with only coordinates.
    :param cube_file: (String) - Cube data file name.
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
    :param df_horizon: (Pandas.DataFrame) - Horizon data frame.
    :param hor_x: (String) - X coordinate column name of horizon data frame.
    :param hor_y: (String) - Y coordinate column name of horizon data frame.
    :param hor_il: (String) - Inline number column name of horizon data frame.
    :param hor_xl: (String) - Cross-line number column name of horizon data frame.
    :param hor_z: (String) - Z coordinate column name of horizon data frame.
    :param value_name: (String) - Value name of data.
    :param match_on: (String) - Default is 'xy'. Options are 'xy' and 'ix'.
                     If 'xy', horizon xy coordinates will be matched with cube xy coordinates to get data from cube
                     to horizon.
                     If 'ix', horizon inline and cross-line number will be matched with cube inline and cross-line
                     number to get data from cube to horizon.
    :param x_win: (Float) - Default is 25.0 when match_on=='xy' and 1.0 when match_on=='ix'.
                  The window in which the horizon x coordinates (or inline numbers) will be matched with the
                  cube x coordinates (or inline numbers).
    :param y_win: (Float) - Default is 25.0 when match_on=='xy' and 1.0 when match_on=='ix'.
                  The window in which the horizon y coordinates (or cross-line numbers) will be matched with the
                  cube y coordinates (or cross-line numbers).
    :param z_win: (Float) - Default is 2.0. The window in which the horizon z coordinates will be matched with the
                  cube z coordinates.
    :return: df_horizon: (Pandas.DataFrame) - Horizon data frame with data from the cube.
    """
    # Load cube data.
    with segyio.open(cube_file) as f:
        f.mmap()
        inline = f.ilines
        xline = f.xlines
        x = np.zeros(len(inline), dtype='float32')
        y = np.zeros(len(xline), dtype='float32')
        for i in range(len(inline)):
            x[i] = f.header[i * len(xline)][header_x] * scl_x
        for i in range(len(xline)):
            y[i] = f.header[i][header_y] * scl_y
        z = f.samples
        cube_data = segyio.tools.cube(f)
        print('Cube info:')
        print('Inline: %d-%d [%dlines]' % (inline[0], inline[-1], len(inline)))
        print('Xline: %d-%d [%dlines]' % (xline[0], xline[-1], len(xline)))
        print('X Range: [%d-%d] [%dsamples]' % (x[0], x[-1], len(x)))
        print('Y Range: [%d-%d] [%dsamples]' % (y[0], y[-1], len(y)))
        print('Z Range: [%d-%d] [%dsamples]' % (z[0], z[-1], len(z)))
    f.close()
    # Get values from cube.
    if match_on == 'xy':
        x_temp = df_horizon[hor_x].drop_duplicates().values
        y_temp = df_horizon[hor_y].drop_duplicates().values
        print('Horizon info:')
        print('X Range: %d-%d [%d samples]' % (x_temp[0], x_temp[-1], len(x_temp)))
        print('Y Range: %d-%d [%d samples]' % (y_temp[0], y_temp[-1], len(y_temp)))
        x_dist = scipy.spatial.distance.cdist(np.reshape(df_horizon[hor_x].values, (-1, 1)),
                                              np.reshape(x, (-1, 1)),
                                              metric='minkowski', p=1)
        y_dist = scipy.spatial.distance.cdist(np.reshape(df_horizon[hor_y].values, (-1, 1)),
                                              np.reshape(y, (-1, 1)),
                                              metric='minkowski', p=1)
    elif match_on == 'ix':
        x_temp = df_horizon[hor_il].drop_duplicates().values
        y_temp = df_horizon[hor_xl].drop_duplicates().values
        print('Horizon info:')
        print('Inline Range: %d-%d [%d samples]' % (x_temp[0], x_temp[-1], len(x_temp)))
        print('Xline Range: %d-%d [%d samples]' % (y_temp[0], y_temp[-1], len(y_temp)))
        x_dist = scipy.spatial.distance.cdist(np.reshape(df_horizon[hor_il].values, (-1, 1)),
                                              np.reshape(inline, (-1, 1)),
                                              metric='minkowski', p=1)
        y_dist = scipy.spatial.distance.cdist(np.reshape(df_horizon[hor_xl].values, (-1, 1)),
                                              np.reshape(xline, (-1, 1)),
                                              metric='minkowski', p=1)
    else:
        raise ValueError("Parameter 'match_on' can only be 'xy' or 'ix'.")
    z_dist = scipy.spatial.distance.cdist(np.reshape(df_horizon[hor_z].values, (-1, 1)),
                                          np.reshape(z, (-1, 1)),
                                          metric='minkowski', p=1)
    indx = np.argmin(x_dist, axis=1)  # Get x indexes of cube data.
    indy = np.argmin(y_dist, axis=1)  # Get y indexes of cube data.
    indz = np.argmin(z_dist, axis=1)  # Get z indexes of cube data.
    if x_win is None:
        if match_on == 'xy':
            x_win = 25.0
        elif match_on == 'ix':
            x_win = 1.0
    if y_win is None:
        if match_on == 'xy':
            y_win = 25.0
        elif match_on == 'ix':
            y_win = 1.0
    x_dist_min = np.amin(x_dist, axis=1)  # Minimum distance of horizon x to cube x.
    y_dist_min = np.amin(y_dist, axis=1)  # Minimum distance of horizon y to cube y.
    z_dist_min = np.amin(z_dist, axis=1)  # Minimum distance of horizon z to cube z.
    # Match horizon and cube coordinates in windows.
    ix = np.squeeze(np.argwhere(x_dist_min < x_win))
    iy = np.squeeze(np.argwhere(y_dist_min < y_win))
    iz = np.squeeze(np.argwhere(z_dist_min < z_win))
    ind = np.intersect1d(np.intersect1d(ix, iy), iz)
    df_horizon.loc[ind, value_name] = cube_data[indx[ind], indy[ind], indz[ind]]
    return df_horizon

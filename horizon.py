import os
import sys
import time
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
    # Get min and max of x, y, inline and xline.
    print('Interpolating horizon depth...')
    if infer:
        if x_col is not None and y_col is not None:
            xmin, xmax = np.nanmin(df[x_col].values), np.nanmax(df[x_col].values)
            ymin, ymax = np.nanmin(df[y_col].values), np.nanmax(df[y_col].values)
            print('\tInferred x range: %.2f-%.2f' % (xmin.item(), xmax.item()))
            print('\tInferred y range: %.2f-%.2f' % (ymin.item(), ymax.item()))
        else:
            raise ValueError('x column or y column name is not defined.')
    else:
        if x_col is not None and y_col is not None:
            xmin, xmax = xy_range[0], xy_range[1]
            ymin, ymax = xy_range[2], xy_range[3]
            print('\tDefined x range: %.2f-%.2f' % (xmin, xmax))
            print('\tDefined y range: %.2f-%.2f' % (ymin, ymax))
        else:
            raise ValueError('x column or y column name is not defined.')
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


def visualize_horizon(df=None, x_name='x', y_name='y', value_name=None, deltax=25.0, deltay=25.0, cmap='seismic_r',
                      vmin=None, vmax=None, nominal=False, class_code=None, class_label=None, fig_name=None, show=True):
    """
    Visualize horizon data.
    :param df: (pandas.DataFrame) - Horizon data frame which contains ['x', 'y', 'value1', 'value2', '...'] columns.
    :param x_name: (String) - Default is 'x'. x-coordinate column name.
    :param y_name: (String) - Default is 'y'. y-coordinate column name.
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
    df_grid = pd.DataFrame({'x': xgrid, 'y': ygrid})
    df_horizon = pd.merge(left=df_grid, right=df, how='left', on=['x', 'y'])  # Merge irregular horizon on regular grid.
    data = df_horizon[value_name].values
    x = df_horizon['x'].values
    y = df_horizon['y'].values
    # Plot horizon.
    plt.figure(figsize=(12, 8))
    if fig_name is None:
        plt.title('Result', fontsize=20)
    else:
        plt.title(fig_name, fontsize=20)
    plt.ticklabel_format(style='plain')
    extent = [np.amin(x), np.amax(x), np.amin(y), np.amax(y)]
    plt.imshow(data.reshape([len(ynew), len(xnew)], order='F'),
               origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
               cmap=plt.cm.get_cmap(cmap), extent=extent)
    if nominal:
        class_code.sort()
        tick_min = (max(class_code) - min(class_code)) / (2 * len(class_code)) + min(class_code)
        tick_max = max(class_code) - (max(class_code) - min(class_code)) / (2 * len(class_code))
        tick_step = (max(class_code) - min(class_code)) / len(class_code)
        ticks = np.arange(start=tick_min, stop=tick_max + tick_step, step=tick_step)
        cbar = plt.colorbar(ticks=ticks)
        cbar.ax.set_yticklabels(class_label)
        cbar.ax.tick_params(axis='y', labelsize=16)
    else:
        plt.colorbar()
    if show:
        plt.show()


def horizon_log(df_horizon=None, df_well_coord=None, log_file_path=None, well_name_col='well_name',
                well_x_col='well_X', well_y_col='well_Y', horizon_x_col='x', horizon_y_col='y', horizon_t_col='t',
                log_t_col='TWT', log_value_col=None, log_abnormal_value=-999, log_file_suffix='.txt',
                print_progress=False):
    """
    Mark log values on horizon.
    :param df_horizon: (pandas.DataFrame) - Horizon data frame which contains ['x', 'y', 't'] columns.
    :param df_well_coord: (pandas.DataFrame) - Well coordinates data frame
                          which contains ['well_name', 'well_x', 'well_y'] columns.
    :param log_file_path: (String) - Time domain well log file directory.
    :param well_name_col: (String) - Default is 'well_name'. Well name column name in df_well_coord.
    :param well_x_col: (String) - Default is 'well_X'. Well x-coordinate column name in df_well_coord.
    :param well_y_col: (String) - Default is 'well_Y'. Well y-coordinate column name in df_well_coord.
    :param horizon_x_col: (String) - Default is 'x'. Horizon x-coordinate column name in df_horizon.
    :param horizon_y_col: (String) - Default is 'y'. Horizon y-coordinate column name in df_horizon.
    :param horizon_t_col: (String) - Default is 't'. Horizon two-way time column name in df_horizon.
    :param log_t_col: (String) - Default is 'TWT'. Well log two-way time column name.
    :param log_value_col: (String) - Well log value column name.
    :param log_abnormal_value: (Float) - Default is -999. Well log abnormal value.
    :param log_file_suffix: (String) - Default is '.txt'. Well log file suffix.
    :param print_progress: (Bool) - Default is False. Whether to print progress.
    :return: df_out: (pandas.DataFrame) - Output data frame which contains ['x', 'y', 't', 'log value', 'well name']
                     columns.
    """
    # Initialize output data frame as a copy of horizon data frame.
    df_out = df_horizon.copy()
    # List of well log files.
    log_file_list = os.listdir(log_file_path)
    # Mark log values on horizon.
    for log_file in log_file_list:
        # Load well log data.
        df_log = pd.read_csv(os.path.join(log_file_path, log_file), delimiter='\t')
        drop_ind = [x for x in range(len(df_log)) if df_log.loc[x, log_value_col] == log_abnormal_value]
        df_log.drop(index=drop_ind, inplace=True)
        df_log.reset_index(drop=True, inplace=True)
        # Get well name.
        well_name = log_file[:-len(log_file_suffix)]
        # Print progress.
        if print_progress:
            sys.stdout.write('\rMatching well %s' % well_name)
        # Get well coordinates from well coordinate file.
        [well_x, well_y] = \
            np.squeeze(df_well_coord[df_well_coord[well_name_col] == well_name][[well_x_col, well_y_col]].values)
        # Get horizon coordinates.
        horizon_x = df_horizon[horizon_x_col].values
        horizon_y = df_horizon[horizon_y_col].values
        # Compute distance map between well and horizon.
        xy_dist = np.sqrt((horizon_x - well_x) ** 2 + (horizon_y - well_y) ** 2)
        # Get array index of minimum distance in distance map. This is the horizon coordinate closest to the well.
        idx_xy = np.argmin(xy_dist)
        # Get horizon two-way time at the closest point to the well.
        horizon_t = df_horizon.loc[idx_xy, horizon_t_col]
        # Get well log two-way time.
        log_t = df_log[log_t_col].values
        # Compute distances between well log two-way time and horizon two-way time at the closest point to the well.
        t_dist = np.abs(log_t - horizon_t)
        # Get array index of the minimum distance. This is the vertically closest point of the well log to the horizon.
        idx_t = np.argmin(t_dist)
        # Get log value.
        log_value = df_log.loc[idx_t, log_value_col]
        # Mark log value on horizon.
        df_out.loc[idx_xy, log_value_col] = log_value
        # Mark well name.
        df_out.loc[idx_xy, 'WellName'] = well_name
    # Drop NaN.
    df_out.dropna(axis='index', how='any', inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    return df_out


def plot_markers(df=None, x_col=None, y_col=None, class_col=None, wellname_col=None, class_label=None, colors=None,
                 annotate=True, anno_color='k', anno_fontsize=12, anno_shift=None):
    """
    Plot markers on horizon.
    :param df: (pandas.Dataframe) - Marker data frame which contains ['x', 'y', 'class', 'well name'] columns.
    :param x_col: (String) - x-coordinate column name.
    :param y_col: (String) - y-coordinate column name.
    :param class_col: (String) - Class column name.
    :param wellname_col: (String) - Well name column name.
    :param class_label: (List of strings) - Label names of the markers.
    :param colors: (List of strings) - Color names of the markers.
    :param annotate: (Bool) - Default is True. Whether to annotate well names beside well markers.
    :param anno_color: (String) - Annotation text color.
    :param anno_fontsize: (Integer) - Default is 12. Annotation text font size.
    :param anno_shift: (List of floats) - Default is [0, 0]. Annotation text coordinates.
    """
    # Get class codes.
    classes = df[class_col].copy()
    classes.drop_duplicates(inplace=True)
    classes = classes.values
    classes.sort()
    # Plot markers.
    for i in range(len(classes)):
        idx = [x for x in range(len(df)) if df.loc[x, class_col] == classes[i]]
        sub_frame = df.loc[idx, [x_col, y_col]]
        x, y = sub_frame.values[:, 0], sub_frame.values[:, 1]
        plt.scatter(x, y, c=colors[classes[i]], edgecolors='k', label=class_label[classes[i]],
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


def FSDI_horizon(df_horizon=None, df_horizon_log=None, coord_col=None, feature_col=None, log_col=None,
                 scale=True, weight=None):
    """
    Feature and Space Distance based Interpolation for horizons.
    :param df_horizon: (pandas.DataFrame) - Horizon data frame, these are points to interpolate, which should
                       contains coordinates columns and features columns.
    :param df_horizon_log: (pandas.DataFrame) - Horizon data frame with marked log values, these are control points,
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
    df_l = df_horizon_log[coord_col + [log_col]].copy()
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
    return df_horizon, df_horizon_log


def FSDI_interhorizon(seis_file=None, seis_name=None,
                      horizon_file=None, horizon_col=None, horizon_x_col='x', horizon_y_col='y', horizon_z_col='t',
                      horizon_dx=25.0, horizon_dy=25.0, horizon_xy_infer=False, horizon_xy_range=None,
                      log_dir=None, log_z_col='TWT', log_value_col=None, log_file_suffix='.txt',
                      well_loc_file=None, well_name_col='well_name', well_x_col='well_X', well_y_col='well_Y',
                      dp=None, weight=None, fill_value=None, init_value=None, tight_frame=True,
                      output_file=None):
    """
    Feature and Space Distance based Interpolation for inter-horizon.
    :param seis_file: (String or list of strings) - Seismic feature. SEG-Y seismic file name with absolute file path.
                      For multiple file, input file name list such as ['.../.../a.sgy', '.../.../b.sgy'].
    :param seis_name: (String or list of strings) - Seismic feature name. Must correspond to seismic files. For multiple
                      seismic feature, input name list such as ['a', 'b'].
    :param horizon_file: (List of strings) - Horizon file list. Horizon file with suffix '.txt' or '.dat' which contains
                         ASCII data and no header. Notice that in each horizon file there must be 3 columns of
                         x, y and t data.
    :param horizon_col: (List of strings) - Define column names of each horizon file.
    :param horizon_x_col: (String) - Default is 'x'. X-coordinate column name of horizon file.
    :param horizon_y_col: (String) - Default is 'y'. Y-coordinate column name of horizon file.
    :param horizon_z_col: (String) - Default is 't'. Z-coordinate column name of horizon file.
    :param horizon_dx: (Float or integer) - Default is 25.0. X-coordinate spacing of every horizon.
    :param horizon_dy: (Float or integer) - Default is 25.0. Y-coordinate spacing of every horizon.
    :param horizon_xy_infer: (Bool) - Default is True. Whether to infer x and y range from horizon data.
    :param horizon_xy_range: (List of floats) - When infer is False, require manually input x and y range
                             [x_min, x_max, y_min, y_max].
    :param log_dir: (String) - Time domain well log file directory.
    :param log_z_col: (String) - Default is 'TWT'. Well log two-way time column name.
    :param log_value_col: (String) - Well log value column name.
    :param log_file_suffix: Default is '.txt'. Well log file suffix.
    :param well_loc_file: (String) - Well location file name.
    :param well_name_col: (String) - Default is 'well_name'. Well name column name in well location file.
    :param well_x_col: (String) - Default is 'well_X'. Well x-coordinate column name in well location file.
    :param well_y_col: (String) - Default is 'well_Y'. Well y-coordinate column name in well location file.
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
    :return: cube_itp: (numpy.3darray) - Interpolation result.
    """
    t1 = time.perf_counter()
    # Read seismic file.
    if isinstance(seis_file, str):
        seis_file = [seis_file]
    if isinstance(seis_file, str) is False and isinstance(seis_file, list) is False:
        raise ValueError('Seismic file must be string or list of strings.')
    if isinstance(seis_name, str):
        if len(seis_file) != 1:
            raise ValueError('The number of seismic names must match the number of seismic files.')
    if isinstance(seis_name, list):
        if len(seis_name) != len(seis_file):
            raise ValueError('The number of seismic names must match the number of seismic files.')
    seis = []  # Initiate list to store seismic data.
    for file in seis_file:
        with segyio.open(file) as f:
            print('Read seismic data from file: ', file)
            # Memory map file for faster reading (especially if file is big...)
            f.mmap()
            # Print file information.
            print('\tFile info:')
            print('\tinline range: %d-%d [%d lines]' % (f.ilines[0], f.ilines[-1], len(f.ilines)))
            print('\tcrossline range: %d-%d [%d lines]' % (f.xlines[0], f.xlines[-1], len(f.xlines)))
            print('\tDepth range: %dms-%dms [%d samples]' % (f.samples[0], f.samples[-1], len(f.samples)))
            dt = segyio.tools.dt(f) / 1000
            print('\tSampling interval: %.1fms' % dt)
            print('\tTotal traces: %d' % f.tracecount)
            # Read seismic data.
            cube = segyio.tools.cube(f)
            # When reading the last file, extract sampling time and coordinates.
            if file == seis_file[-1]:
                # Read sampling time.
                t = f.samples
                # Extract trace coordinates from trace header.
                x = np.zeros(shape=(f.tracecount,), dtype='float32')
                y = np.zeros(shape=(f.tracecount,), dtype='float32')
                for i in range(f.tracecount):
                    sys.stdout.write('\rExtracting trace coordinates: %.2f%%' % ((i + 1) / f.tracecount * 100))
                    x[i] = f.header[i][73] * 1e-1  # Adjust coordinate according to actual condition.
                    y[i] = f.header[i][77] * 1e-1  # Adjust coordinate according to actual condition.
                sys.stdout.write('\n')
                # Re-shape the trace coordinates array to match the seismic data cube.
                x_2d = x.reshape([len(f.ilines), len(f.xlines)], order='C')
                y_2d = y.reshape([len(f.ilines), len(f.xlines)], order='C')
                # The corresponding coordinates index.
                x_ind = np.linspace(0, len(f.ilines)-1, len(f.ilines), dtype='int32')
                y_ind = np.linspace(0, len(f.xlines)-1, len(f.xlines), dtype='int32')
                x_ind, y_ind = np.meshgrid(x_ind, y_ind, indexing='ij')
                x_ind = np.ravel(x_ind, order='C')
                y_ind = np.ravel(y_ind, order='C')
        f.close()
        seis.append(cube)
    # Read horizon files.
    if isinstance(horizon_file, list) is False:
        raise ValueError('Horizon file must be list of strings.')
    if isinstance(horizon_file, list) and len(horizon_file) == 1:
        raise ValueError('The number of horizons must >=2. If you want to process single horizon, '
                         'use function FSDI_horizon.')
    horizon = []  # Initiate list to store horizon data.
    for file in horizon_file:
        sys.stdout.write('Reading horizon file %s...' % file)
        df_horizon = pd.read_csv(file, names=horizon_col, delimiter='\t')
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
            df_horizon = horizon_interp(df_horizon, x_col=horizon_x_col, y_col=horizon_y_col, t_col=horizon_z_col,
                                        x_step=horizon_dx, y_step=horizon_dy, visualize=False, infer=horizon_xy_infer,
                                        xy_range=horizon_xy_range)
            if np.equal(df_horizon[horizon_x_col].values, x).all() and \
                    np.equal(df_horizon[horizon_y_col].values, y).all():
                raise ValueError('Horizon x and y coordinates cannot match seismic, require manually input xy range.\n')
        horizon.append(df_horizon)
    # Generate pseudo-inner-horizons.
    horizon_new = []
    z_ind_list = []
    z_ind_min = 1e30
    z_ind_max = -999
    for i in range(len(horizon) - 1):
        # Compute thickness between horizons.
        th = np.abs(horizon[i+1][horizon_z_col].values - horizon[i][horizon_z_col].values)
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
            # Add seismic features to pseudo-inner-horizons.
            dist_map = scipy.spatial.distance.cdist(np.reshape(df_temp[horizon_z_col].values, (-1, 1)),
                                                    np.reshape(t, (-1, 1)), metric='minkowski', p=1)
            z_ind = np.argmin(dist_map, axis=1)  # The horizon depth indexes of every trace in seismic cubes.
            if np.amin(z_ind) < z_ind_min:
                z_ind_min = np.amin(z_ind)
            if np.amax(z_ind) > z_ind_max:
                z_ind_max = np.amax(z_ind)
            z_ind_list.append(z_ind)
            for k in range(len(seis)):
                if len(seis) == 1:
                    df_temp[seis_name] = seis[k][x_ind, y_ind, z_ind]
                else:
                    df_temp[seis_name[k]] = seis[k][x_ind, y_ind, z_ind]
            horizon_new.append(df_temp)
    sys.stdout.write('\n')
    # Read well location file.
    df_well_loc = pd.read_csv(well_loc_file, delimiter='\s+')
    # Initiate interpolation result array.
    if tight_frame:  # Tight box containing all horizons.
        t = t[z_ind_min: z_ind_max + 1]
        cube_itp = np.ones([seis[0].shape[0], seis[0].shape[1], len(t)], dtype='float32') * fill_value
        np.savetxt(output_file[:-4] + '_depth.txt', t.reshape(-1, 1))
        for i in range(len(x_ind)):
            sys.stdout.write('\rInitiating array for FSDI interpolation:%.2f%%' % ((i+1) / len(x_ind) * 100))
            cube_itp[x_ind[i], y_ind[i], z_ind_list[0][i]-z_ind_min:z_ind_list[-1][i]+1-z_ind_min] = init_value
        sys.stdout.write('\n')
        print('\tTight frame: ', tight_frame)
        print('\tDepth range: %dms-%dms [%d samples]' % (t[0], t[-1], int((t[-1] - t[0]) / dt + 1)))
    else:
        cube_itp = np.ones(seis[0].shape, dtype='float32') * fill_value
        np.savetxt(output_file[:-4] + '_depth.txt', t.reshape(-1, 1))
        for i in range(len(x_ind)):
            sys.stdout.write('\rInitiating array for FSDI interpolation:%.2f%%' % ((i+1) / len(x_ind) * 100))
            cube_itp[x_ind[i], y_ind[i], z_ind_list[0][i]:z_ind_list[-1][i]+1] = init_value
        sys.stdout.write('\n')
        print('\tTight frame:', tight_frame)
        print('\tDepth range: %dms-%dms [%d samples]' % (t[0], t[-1], int((t[-1] - t[0]) / dt + 1)))
    # FSDI on pseudo-inner-horizons.
    for i in range(len(horizon_new)):
        sys.stdout.write('\rInterpolating: %.2f%% [%d/%d]' % ((i+1)/len(horizon_new)*100, i+1, len(horizon_new)))
        df_ctp = horizon_log(df_horizon=horizon_new[i], df_well_coord=df_well_loc, log_file_path=log_dir,
                             well_name_col=well_name_col, well_x_col=well_x_col, well_y_col=well_y_col,
                             horizon_x_col=horizon_x_col, horizon_y_col=horizon_y_col, horizon_t_col=horizon_z_col,
                             log_t_col=log_z_col, log_value_col=log_value_col, log_file_suffix=log_file_suffix)
        df_interp, _ = FSDI_horizon(df_horizon=horizon_new[i], df_horizon_log=df_ctp,
                                    coord_col=[horizon_x_col, horizon_y_col, horizon_z_col], feature_col=seis_name,
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
    sys.stdout.write('Saving interpolation result to file %s...' % output_file)
    cube_out = np.reshape(cube_itp, (cube_itp.shape[0] * cube_itp.shape[1], cube_itp.shape[2]), order='C')
    np.savetxt(output_file, np.c_[x, y, cube_out], delimiter='\t')
    sys.stdout.write(' Done.\n')
    t2 = time.perf_counter()
    # Print process time.
    print('Process time: %.2fs' % (t2 - t1))
    return cube_itp

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn import preprocessing
from matplotlib.colors import LinearSegmentedColormap


def horizon_interp(df=None, x_step=25.0, y_step=25.0, inline_step=1, xline_step=1, method='linear', visualize=True):
    """
    Interpolate horizon on a regular grid.
    :param df: (pandas.DataFrame) - Horizon data frame which contains ['inline', 'xline', 'x', 'y', 't'] columns.
    :param x_step: (Float) - Default is 25.0. x coordinate step of the regular grid.
    :param y_step: (Float) - Default is 25.0. y coordinate step of the regular grid.
    :param inline_step: (Integer) - Default is 1. Inline step.
    :param xline_step: (Integer) - Default is 1. Cross-line step.
    :param method: (String) - Default is 'linear'. Method of interpolation. One of {'linear', 'nearest', 'cubic'}.
    :param visualize: (Bool) - Default is True. Whether to visualize the interpolation result.
    :return df_new: (pandas.DataFrame) - Interpolated horizon data frame.
    """
    # Check NaN.
    nan_presence = df.isna().any().any()
    nan_number = df.isna().sum().sum()
    print('NaN presence:', nan_presence)
    print('Total number of NaN:', nan_number)
    # Drop rows which contain NaN.
    if nan_presence is True:
        print('Drop rows which contain NaN...')
        df.dropna(axis='index', how='any', inplace=True)
        print('Complete.')
    # Get min and max of x, y, inline and xline.
    xmin, ymin = df.min(axis=0)['x'], df.min(axis=0)['y']
    xmax, ymax = df.max(axis=0)['x'], df.max(axis=0)['y']
    inline_min, xline_min = df.min(axis=0)['inline'], df.min(axis=0)['xline']
    inline_max, xline_max = df.max(axis=0)['inline'], df.max(axis=0)['xline']
    # Get 2D coordinates of control points.
    coord = df[['x', 'y']].values
    t = df['t'].values
    # Create new 2D coordinates to interpolate.
    xnew = np.linspace(xmin, xmax, int((xmax - xmin) / x_step) + 1, dtype='float32')
    ynew = np.linspace(ymin, ymax, int((ymax - ymin) / y_step) + 1, dtype='float32')
    inline_new = np.linspace(inline_min, inline_max, int((inline_max - inline_min) / inline_step) + 1,
                             dtype='int32')
    xline_new = np.linspace(xline_min, xline_max, int((xline_max - xline_min) / xline_step) + 1, dtype='int32')
    xnew, ynew = np.meshgrid(xnew, ynew)
    inline_new, xline_new = np.meshgrid(inline_new, xline_new)
    # Interpolate.
    tnew_linear = interpolate.griddata(points=coord, values=t, xi=(xnew, ynew), method=method)
    if visualize:
        # Visualize.
        plt.figure()
        plt.title('Interpolation Result')
        cset = plt.contourf(xnew, ynew, tnew_linear, 8, cmap='rainbow')
        plt.colorbar(cset)
        plt.show()
    # Save horizon data.
    horizon_new = np.c_[inline_new.ravel(order='F'), xline_new.ravel(order='F'),
                        xnew.ravel(order='F'), ynew.ravel(order='F'), tnew_linear.ravel(order='F')]
    horizon_new = horizon_new.astype('float32')
    df_new = pd.DataFrame(horizon_new, columns=df.columns)
    df_new[['inline', 'xline']] = df_new[['inline', 'xline']].astype('int32')
    print('Done.')
    return df_new


def visualize_horizon(df=None, x_name='x', y_name='y', value_name=None, deltax=25.0, deltay=25.0, cmap='seismic_r',
                      vmin=None, vmax=None, nominal=False, class_code=None, class_label=None, fig_name=None, show=True,
                      save_fig=False, save_path=None, save_file=None):
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
    :param save_fig: (Bool) - Default is False. Whether to save the figure.
    :param save_path: (String) - File directory to save the figure.
    :param save_file: (String) - File name to save the figure.
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
    if save_fig:
        if save_file is None:
            save_file = fig_name
        plt.savefig(os.path.join(save_path, save_file + '.png'), dpi=200)
    if show:
        plt.show()


def horizon_log(df_horizon=None, df_well_coord=None, log_file_path=None, well_name_col='well_name',
                well_x_col='well_X', well_y_col='well_Y', horizon_x_col='x', horizon_y_col='y', horizon_t_col='t',
                log_t_col='TWT', log_value_col=None, abnormal=-999, log_file_suffix='.txt'):
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
    :param abnormal: (Integer or Float) - Default is -999. Replace NaN value in output data frame.
    :param log_file_suffix: (String) - Default is '.txt'. Well log file suffix.
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
        # Get well name.
        well_name = log_file[:-len(log_file_suffix)]
        # Print progress.
        sys.stdout.write('\r\tMatching well %s' % well_name)
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
    # Fill NaN with defined value.
    df_out.fillna(abnormal, inplace=True)
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


def FSDinterp(df_horizon=None, df_horizon_log=None, coord_col=None, feature_col=None, log_col=None, scale=True):
    """
    Feature and Space Distance based well log interpolation.
    :param df_horizon: (pandas.DataFrame) - Horizon data frame, these are points to interpolate, which should
                       contains coordinates columns and features columns.
    :param df_horizon_log: (pandas.DataFrame) - Horizon data frame with marked log values, these are control points,
                           which should contains coordinates columns and log columns.
    :param coord_col: (List of Strings) - Coordinates columns names. (e.g. ['x', 'y', 't']).
    :param feature_col: (String or list of strings) - Features columns names.
    :param log_col: (String) - Well log column name.
    :param scale: (Bool) - Default is True. Whether to scale coordinates and features.
    :return: df_horizon: (pandas.DataFrame) - Interpolation result.
             df_horizon_log: (pandas.DataFrame) - Control points.
    """
    # Make a copy of horizon data frame.
    df_horizon_copy = df_horizon.copy()
    # Match features to control points by coordinates.
    df_horizon_log = pd.merge(left=df_horizon_log, right=df_horizon_copy, on=coord_col, how='inner')
    # Make a copy of horizon with log value data frame.
    df_horizon_log_copy = df_horizon_log.copy()
    # Assign target columns.
    if isinstance(feature_col, str):
        feature_col = [feature_col]
    column = coord_col + feature_col
    # Whether to scale coordinates and features.
    if scale:
        # MinMaxScalar.
        min_max_scalar = preprocessing.MinMaxScaler()
        # Transform (Scale).
        df_horizon_copy[column] = min_max_scalar.fit_transform(df_horizon_copy[column].values)  # Points to interpolate.
        df_horizon_log_copy[column] = min_max_scalar.transform(df_horizon_log_copy[column].values)  # Control points.
    # Get control points' coordinates and features.
    control = df_horizon_log_copy[column].values
    # Get control points' log values.
    control_log = df_horizon_log_copy[log_col].values
    # Get horizon's coordinates and features.
    horizon = df_horizon_copy[column].values
    # Compute feature & space distance map.
    dist_map = np.zeros([len(horizon), len(control)], dtype=np.float32)
    for i in range(len(control)):
        dist = np.sqrt(np.sum((horizon - control[i, :]) ** 2, axis=1))  # Distance map of well i.
        dist_map[:, i] = dist
    # Get column index of minimum distance.
    min_idx = np.argmin(dist_map, axis=1)
    # Interpolate log values according to minimum distance.
    df_horizon[log_col] = control_log[min_idx]
    return df_horizon, df_horizon_log


# Main function.
if __name__ == '__main__':
    # Format.
    pd.options.display.max_rows = 100
    pd.options.display.max_columns = 20
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    # Directory.
    base_path = 'D:/Opendtect/Database/Niuzhuang/'
    horizon_file_path = 'Export'
    horizon_log_file_path = 'HorizonsLog'
    output_file_path = 'E:/科研项目/多学科异构数据整合与智能建模方法/图片'
    # Parameters.
    hl_keyword = '_dense'  # Horizon with log file keyword.
    h_keyword = '-features'  # Horizon with features file keyword.
    horizon = 'z6'
    feature_name = ['seismic', 'sp', 'vpvs']
    file_format = '.dat'
    horizon_col = ['x', 'y', 't'] + feature_name
    log_name = 'Litho_Code'  # Lithology.
    well_name = 'WellName'
    feature_abnormal_value = 1e30
    log_abnormal_value = -999
    binarize = False
    save_fig = False
    # Read horizon data with features.
    df_horizon = pd.read_csv(os.path.join(base_path + horizon_file_path, horizon + h_keyword + file_format),
                             header=None, skiprows=2, names=horizon_col, delimiter='\t')
    # Drop rows with abnormal value.
    df_horizon.mask(df_horizon == feature_abnormal_value, inplace=True)
    df_horizon.dropna(axis=0, how='any', inplace=True)
    df_horizon.reset_index(drop=True, inplace=True)
    # Adjust float decimal (2 decimal places).
    df_horizon['t'] = df_horizon['t'].round(2)
    # Read horizon data with log markers.
    df_horizon_log = pd.read_csv(os.path.join(base_path + horizon_log_file_path, horizon + hl_keyword + file_format),
                                 delimiter='\t')
    # Whether to binarize lithology.
    if binarize:
        # Convert to binary lithology.
        df_horizon_log.loc[df_horizon_log[log_name] == 1, log_name] = 0  # Convert to binary.
        df_horizon_log.loc[df_horizon_log[log_name] > 1, log_name] = 1  # Convert to binary.
    # Make markers.
    drop_idx = [idx for idx in range(len(df_horizon_log)) if df_horizon_log.loc[idx, log_name] == log_abnormal_value]
    marker = df_horizon_log.drop(drop_idx)
    marker.reset_index(inplace=True, drop=True)
    # Feature and Space Distance based Interpolation (FSDI).
    df_horizon, marker = FSDinterp(df_horizon=df_horizon, df_horizon_log=marker, coord_col=['x', 'y', 't'],
                                   log_col=log_name, feature_col=feature_name)
    # Marker information.
    if binarize:
        marker_color = ['grey', 'gold']  # Binary lithology.
        labels = ['泥岩', '砂岩']  # Binary lithology.
        class_code = [0, 1]  # Lithology codes.
        cm = LinearSegmentedColormap.from_list('custom', marker_color, len(marker_color))
    else:
        marker_color = ['grey', 'limegreen', 'cyan', 'gold', 'darkviolet']  # Five lithology.
        labels = ['泥岩', '灰质泥岩', '粉砂岩', '砂岩', '含砾砂岩']  # Five lithology.
        class_code = [0, 1, 2, 3, 4]  # Lithology codes.
        cm = LinearSegmentedColormap.from_list('custom', marker_color, len(marker_color))
    for i in range(len(feature_name)):
        # Visualize horizon data.
        visualize_horizon(df=df_horizon, value_name=feature_name[i], show=False, cmap='seismic',
                          fig_name=horizon + '-' + feature_name[i])
        # Plot markers.
        plot_markers(marker, x_col='x', y_col='y', class_col=log_name, class_label=labels, colors=marker_color,
                     wellname_col=well_name)
        # Add legend.
        plt.legend(loc='upper right', fontsize=15)
        # Save figure.
        if save_fig:
            if binarize:
                plt.savefig(os.path.join(output_file_path, horizon + '-' + feature_name[i] + '(BinaryLithology).png'),
                            dpi=200)
            else:
                plt.savefig(os.path.join(output_file_path, horizon + '-' + feature_name[i] + '(PentaLithology).png'),
                            dpi=200)
    # Visualize interpolation result.
    if binarize:
        visualize_horizon(df=df_horizon, value_name=log_name, show=False, cmap=cm, fig_name=horizon + '-Lithology',
                          nominal=True, class_code=class_code, class_label=labels,
                          vmin=min(class_code), vmax=max(class_code))
    else:
        visualize_horizon(df=df_horizon, value_name=log_name, show=False, cmap=cm, fig_name=horizon + '-Lithology',
                          nominal=True, class_code=class_code, class_label=labels,
                          vmin=min(class_code), vmax=max(class_code))
    # Plot markers.
    plot_markers(marker, x_col='x', y_col='y', class_col=log_name, class_label=labels, colors=marker_color,
                 wellname_col=well_name)
    # Add legend.
    plt.legend(loc='upper right', fontsize=15)
    # Save figure.
    if save_fig:
        if binarize:
            plt.savefig(os.path.join(output_file_path, horizon + '-BinaryLithology.png'), dpi=200)
        else:
            plt.savefig(os.path.join(output_file_path, horizon + '-PentaLithology.png'), dpi=200)
    plt.show()

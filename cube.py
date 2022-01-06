import numpy as np
import pandas as pd
import pyvista as pv
import segyio
import time
import sys
import os
from well_log import resample_log
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from pyvistaqt import BackgroundPlotter


def FSDI_cube(feature_file=None, feature_name=None, header_x=73, header_y=77, scl_x=1, scl_y=1, scale=True, weight=None,
              log_dir=None, log_file_suffix='.txt', vertical_well=True, log_name=None, depth_name=None, coord_name=None,
              abnormal_value=None, resample_method=None, well_location_file=None, well_name_loc=None,
              output_file=None, output_file_suffix='.dat'):
    """
    Feature and distance based interpolation (FSDI) for cubes.
    :param feature_file: (Strings or list of strings) - Feature file name (segy or sgy format).
                         For single file, directly enter file name.
                         For multiple files, enter file names as list of strings, e.g. ['a.sgy', 'b.sgy'].
    :param feature_name: (String or list of strings) - Feature name.
                         For single attribute, directly enter attribute name like 'amplitude'.
                         For multiple attributes, enter attribute names as list of strings, e.g. ['amplitude', 'phase'].
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
    :param scale: (Bool) - Default is True (recommended). Whether to scale coordinates and features to 0 and 1 with
                  MinMaxScalar before distance calculation.
    :param weight: (List of floats) - Default is that all features (including spatial coordinates) have equal weight.
                                      Weight of spatial coordinates and features, e.g. [1, 1, 1, 2, 2] for
                                      ['x', 'y', 'z', 'amplitude', 'vp'].
    :param log_dir: (String) - Time domain well log file directory (folder).
    :param log_file_suffix: (String) - Default is '.txt'. Suffix of log files in log_dir folder.
    :param vertical_well: (Bool) - Whether the wells are vertical.
                          If True, will process wells as vertical wells, the well log files must have columns:
                          depth (depth_name) and log (log_name), and well location file is required.
                          If False, will process wells as inclined wells, the well log files must
                          have columns: x (coord_name[0]), y (coord_name[1]), depth (depth_name) and log (log_name).
    :param log_name: (String or list of strings) - Well log name.
                     For single log, directly enter log name like 'porosity'.
                     For multiple logs, enter log names as list of strings, e.g. ['Litho_Code', 'porosity'].
    :param depth_name: (String) - Well log depth column name.
    :param coord_name: (List of strings) - Well coordinate column names, e.g. ['X', 'Y'].
    :param abnormal_value: (Float) - Default is None, which means no abnormal values in well logs.
                            The abnormal value in log column.
    :param resample_method: (Strings) - Default is None, which is not to resample well logs.
                            Well log re-sampling method, it will resample the time domain well logs to the sampling
                            interval of feature data.
                            Optional methods: 'nearest', 'average', 'median', 'rms', 'most_frequent'.
    :param well_location_file: (String) - Well location file name. Only used when vertical_well=True.
                               This file must contain well name column and XY coordinate columns.
                               The well names must match the well log file name before file suffix (e.g. '.txt').
    :param well_name_loc: (String) - Well name column name in well location file. Only used when vertical_well=True.
    :param output_file: (String or list of strings) - Output file name (ASCII) for interpolation results.
                        For single file, directly enter file name.
                        For multiple files, enter file names as list of strings, e.g. ['a.txt', 'b.txt'].
                        Note that the number of output files should match with the number of logs.
    :param output_file_suffix: (String) - Default is '.dat'. The suffix of the output file.
    :return: cube_itp: (numpy.ndarray) - A 4d array contains the interpolation results.
                       cube_itp[inline, xline, samples, interp_logs].
    """
    t1 = time.perf_counter()  # Timer.
    # Read feature file.
    if isinstance(feature_file, str):
        Nfile = 1
        feature_file = [feature_file]
    elif isinstance(feature_file, list):
        Nfile = len(feature_file)
    else:
        raise ValueError('Feature file must be string or list of strings.')
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
            print('Time range: %dms-%dms [%d samples]' % (f.samples[0], f.samples[-1], len(f.samples)))
            dt = segyio.tools.dt(f) / 1000
            print('Sampling interval: %.1fms' % dt)
            print('Total traces: %d' % f.tracecount)
            # Read feature data.
            cube = segyio.tools.cube(f)
            if file == feature_file[-1]:
                # Read sampling time.
                t = f.samples
                # Extract trace coordinates from trace header.
                x = np.zeros(shape=(f.tracecount, ), dtype='float32')
                y = np.zeros(shape=(f.tracecount, ), dtype='float32')
                for i in range(f.tracecount):
                    sys.stdout.write('\rExtracting trace coordinates: %.2f%%' % ((i+1) / f.tracecount * 100))
                    x[i] = f.header[i][header_x] * scl_x
                    y[i] = f.header[i][header_y] * scl_y
                sys.stdout.write('\n')
                # Re-shape the trace coordinates array to match the feature data cube.
                x = x.reshape([len(f.ilines), len(f.xlines)], order='C')
                y = y.reshape([len(f.ilines), len(f.xlines)], order='C')
                print('X range: %.2f-%.2f' % (np.amin(x), np.amax(x)))
                print('Y range: %.2f-%.2f' % (np.amin(y), np.amax(y)))
        f.close()
        feature.append(cube)
    # Read well log file.
    log_list = os.listdir(log_dir)  # Well log file list.
    # Initiate control points data frame.
    df_ctp = pd.DataFrame()
    if well_location_file is not None:
        # Read well locations from the well location file.
        df_loc = pd.read_csv(well_location_file, delimiter='\s+')
    well_in_area = []
    sys.stdout.write('Assembling well logs...')
    for log_file in log_list:
        df = pd.read_csv(os.path.join(log_dir, log_file), delimiter='\t')
        # Drop rows with abnormal value.
        if abnormal_value is not None:
            drop_col = [x for x in range(len(df)) if df.loc[x, log_name] == abnormal_value]
            df.drop(index=drop_col, inplace=True)
            df.reset_index(drop=True, inplace=True)
        # Re-sample well log by seismic sampling interval.
        if resample_method is not None:
            df = resample_log(df, delta=dt, depth_col=depth_name, log_col=log_name, method=resample_method)
        # Change well coordinates to their nearest cube data coordinates.
        cube_coord = np.c_[x.ravel(), y.ravel()]  # cube data coordinates.
        if vertical_well:  # For vertical well.
            if well_location_file is not None:  # Get well coordinates from well location file.
                well_coord = df_loc.loc[df_loc[well_name_loc] == log_file[:-len(log_file_suffix)],
                                        coord_name].values  # 2d array.
            else:  # Get well coordinates from coordinate columns in well log data frame.
                well_coord = df[coord_name].values[0]  # Coordinates at each rows are the same, get the first one.
            if np.squeeze(well_coord)[0] > np.amax(x) or np.squeeze(well_coord)[0] < np.amin(x) or \
                    np.squeeze(well_coord)[1] > np.amax(y) or np.squeeze(well_coord)[1] < np.amin(y):
                continue  # This well is not in target area, skip to the next well.
            else:
                well_in_area.append(log_file[:-len(log_file_suffix)])  # Record wells in target area.
            xy_dist = np.sqrt(np.sum((well_coord - cube_coord) ** 2, axis=1))
            ind = np.argmin(xy_dist)
            well_coord = cube_coord[ind]
            well_coord = np.ones(shape=[len(df), 2]) * well_coord
            # Add well coordinate to data frame.
            data = np.c_[well_coord, df.values]
            if isinstance(log_name, str):
                df = pd.DataFrame(data=data, columns=coord_name + [depth_name, log_name], copy=True)
            elif isinstance(log_name, list):
                df = pd.DataFrame(data=data, columns=coord_name + [depth_name] + log_name, copy=True)
            else:
                raise ValueError("'log_name' can only be a string or list of strings.")
        else:  # For inclined well.
            flag = 0
            for i in range(len(df)):
                sys.stdout.write('Changing inclined well log coordinates to nearest cube data coordinates: %.2f%%' %
                                 ((i+1) / len(df)))
                log_coord = df.loc[i, coord_name].values  # 1d array.
                if log_coord[0] > np.amax(x) or log_coord[0] < np.amin(x) or \
                        log_coord[1] > np.amax(y) or log_coord[1] < np.amin(y):
                    continue  # This log sample's location is not in target area.
                else:
                    if flag == 0:
                        well_in_area.append(log_file[:-len(log_file_suffix)])  # Record wells in target area.
                        flag = 1
                xy_dist = np.sqrt(np.sum((log_coord - cube_coord) ** 2, axis=1))
                ind = np.argmin(xy_dist)
                log_coord = cube_coord[ind]
                df.loc[i, coord_name] = log_coord
            sys.stdout.write('\n')
        # Add features at control points (well log) to data frame.
        feature_ctp = np.zeros(shape=[len(df), Nfile], dtype='float32')
        if vertical_well:  # Vertical well.
            indx, indy = np.squeeze(np.argwhere((x == well_coord[0, 0]) & (y == well_coord[0, 1])))
            indz0 = np.squeeze(np.argwhere(t == df[depth_name].min()))
            indz1 = np.squeeze(np.argwhere(t == df[depth_name].max()))
            for i in range(Nfile):
                feature_ctp[:, i] = feature[i][indx, indy, indz0:indz1+1]
        else:  # Inclined well.
            for i in range(len(df)):
                indx, indy = np.squeeze(np.argwhere((x == df.loc[i, coord_name[0]]) & (y == df.loc[i, coord_name[1]])))
                indz = np.squeeze(np.argwhere(t == df.loc[i, depth_name]))
                for j in range(Nfile):
                    feature_ctp[i, j] = feature[j][indx, indy, indz]
        df[feature_name] = feature_ctp
        df_ctp = df_ctp.append(df, ignore_index=True)
    sys.stdout.write(' Done.\n')
    if len(well_in_area) == 0:
        raise ValueError('No well in target area, please check well coordinates and cube data coordinates.')
    # FSDInterpolation.
    if weight is None:
        weight = np.ones(Nfile + 3)  # Equal weight of all features.
    else:
        weight = np.array(weight)  # Customized feature weight.
    # Scale.
    if scale:
        # MinMaxScalar.
        xy_scalar = MinMaxScaler()  # x and y scalar.
        t_scalar = MinMaxScaler()  # Two-way time scalar.
        feature_scalar = MinMaxScaler()  # Feature scalar.
        # Fit.
        xy_scalar.fit(np.c_[x.ravel(order='C'), y.ravel(order='C')])
        t_scalar.fit(t.reshape(-1, 1))
        feature_r = np.zeros(shape=[len(feature[0].ravel(order='C')), Nfile], dtype='float32')
        for i in range(Nfile):
            feature_r[:, i] = feature[i].ravel(order='C')
        feature_scalar.fit(feature_r)
    # Determine interpolation depth range by well log depth range.
    depth_min = df_ctp[depth_name].min()
    depth_max = df_ctp[depth_name].max()
    # Initiate interpolation 3d array (this will be the result).
    ind_min = np.squeeze(np.argwhere(t == depth_min))
    ind_max = np.squeeze(np.argwhere(t == depth_max))
    t_cut = t[ind_min: ind_max+1]
    if output_file is not None:
        sys.stdout.write(f'Saving interpolation depth samples to {output_file[:-len(output_file_suffix)]}_depth.txt...')
        np.savetxt(f'{output_file[:-len(output_file_suffix)]}_depth.txt', t_cut.reshape(-1, 1))
        sys.stdout.write(' Done.\n')
    if isinstance(log_name, list):
        cube_itp = np.zeros(shape=[feature[0].shape[0], feature[0].shape[1], len(t_cut), len(log_name)],
                            dtype='float32')
    elif isinstance(log_name, str):
        cube_itp = np.zeros(shape=[feature[0].shape[0], feature[0].shape[1], len(t_cut), 1], dtype='float32')
    else:
        raise ValueError('Log column name can only be string or list of strings.')
    # Get control points.
    if isinstance(feature_name, str):
        feature_name = [feature_name]
    ctp = df_ctp[coord_name + [depth_name] + feature_name].values
    if scale:
        ctp[:, 0:2] = xy_scalar.transform(ctp[:, 0:2])  # Scale control points' x and y coordinates.
        ctp[:, 2] = np.squeeze(t_scalar.transform(ctp[:, 2].reshape(-1, 1)))  # Scale control points' TWT coordinates.
        if ctp[:, 3:].ndim == 1:
            ctp[:, 3:] = np.squeeze(feature_scalar.transform(ctp[:, 3:].reshape(-1, 1)))
        else:
            ctp[:, 3:] = feature_scalar.transform(ctp[:, 3:])  # Scale control points' features.
    # Compute distance map trace by trace.
    for i in range(cube_itp.shape[0]):  # Inlines.
        for j in range(cube_itp.shape[1]):  # Trace number.
            sys.stdout.write('\rInterpolating trace %d/%d [%.2f%%]' %
                             (i*cube_itp.shape[1]+j+1, cube_itp.shape[0]*cube_itp.shape[1],
                              (i*cube_itp.shape[1]+j+1) / (cube_itp.shape[0]*cube_itp.shape[1]) * 100))
            x_itp = x[i, j] * np.ones(cube_itp.shape[2], dtype='float32')  # x-coordinate.
            y_itp = y[i, j] * np.ones(cube_itp.shape[2], dtype='float32')  # y-coordinate.
            t_itp = t_cut.copy()  # Two-way time.
            feature_itp = np.zeros(shape=[cube_itp.shape[2], Nfile], dtype='float32')  # Features.
            for n in range(Nfile):
                feature_itp[:, n] = feature[n][i, j, ind_min: ind_max+1]
            itp = np.c_[x_itp, y_itp, t_itp, feature_itp]  # Points to interpolate.
            if scale:
                itp[:, 0:2] = xy_scalar.transform(itp[:, 0:2])  # Scale interpolate points' x and y coordinates.
                itp[:, 2] = np.squeeze(t_scalar.transform(itp[:, 2].reshape(-1, 1)))  # Scale interpolate points' TWT.
                if itp[:, 3:].ndim == 1:
                    itp[:, 3:] = np.squeeze(feature_scalar.transform(itp[:, 3:].reshape(-1, 1)))
                else:
                    itp[:, 3:] = feature_scalar.transform(itp[:, 3:])  # Scale interpolate points' features.
            dist_map = cdist(itp, ctp, metric='minkowski', w=weight, p=2)  # Compute distance map.
            # Get column index of minimum distance.
            min_idx = np.argmin(dist_map, axis=1)
            # Interpolate log values according to minimum distance.
            log_itp = df_ctp.loc[min_idx, log_name].values
            if log_itp.ndim == 1:
                log_itp = log_itp.reshape(-1, 1)
            cube_itp[i, j, :, :] = log_itp  # Write trace to cube.
    sys.stdout.write('\n')
    # Save interpolation result.
    if output_file is not None:
        x_out = x.ravel(order='C')
        y_out = y.ravel(order='C')
        if isinstance(output_file, str):
            output_file = [output_file]
        for i in range(cube_itp.shape[3]):
            sys.stdout.write('Saving interpolation result to file %s...' % output_file[i])
            cube_out = cube_itp[:, :, :, i].reshape([cube_itp.shape[0] * cube_itp.shape[1], cube_itp.shape[2]],
                                                    order='C')
            np.savetxt(output_file[i], np.c_[x_out, y_out, cube_out], delimiter='\t')
            sys.stdout.write('Done.\n')
    t2 = time.perf_counter()
    print('Process time: %.2fs' % (t2 - t1))
    return cube_itp


def plot_cube(p=None, cube_file=None, header_x=73, header_y=77, scl_x=1, scl_y=1, cube_data=None, value_name=None,
              fig_name=None, colormap='seismic', on='xy', scale=None,
              hor_list=None, hor_sep='\t', hor_header=None, hor_col_names=None,
              hor_x=None, hor_y=None, hor_il=None, hor_xl=None, hor_z=None, hor_xwin=None, hor_ywin=None, hor_zwin=2.0,
              show_slice=True, show_axes=True):
    """
    Visualize cube data and horizon data in the same time.
    :param p: (pyvista.Plotter) - Default is None, which is to create a new plotter. Can also accept a plotter from
              outside.
    :param cube_file: (String) - SEG-Y data file name. If a file name is passed to this parameter, the function will use
                      the data in the file.
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
    :param cube_data: (Numpy.3darray) - The cube data. If a 3D array is passed to this parameter, the function will use
                      this 3D array.
    :param value_name: (String) - Value name of the data.
    :param fig_name: (String) - Figure name.
    :param colormap: (String) - Default is 'seismic'. The color map used to visualize the cube data.
    :param on: (String) - Default is 'xy'. Options are 'xy' and 'ix'.
                          If 'xy', the function will use trace coordinates as xy coordinates.
                          If 'ix', the function will use inline and cross-line numbers as xy coordinates.
                          If input 3D array, this parameter will be forced to be 'xy' and the xy coordinates are
                          automatically generated from data shape.
    :param scale: (List of floats) - Default is None, which is not to scale the coordinates. If scaling is needed, input
                  a length 3 list to control the scale of xyz coordinates. E.g.[10, 10, 1] means make x and y
                  coordinates 10 times larger while z coordinates remain unchanged.
                  Generally, when loading data from SEG-Y files, the on='xy' will make the xy coordinates way larger
                  than z coordinates, thus requiring to scale the x and y coordinates smaller like [0.1, 0.1, 1].
                  On the contrary, on='ix' will make the xy coordinates smaller than z coordinates, thus requiring to
                  scale the xy coordinates larger like [10, 10, 1].
    :param hor_list: (List of strings) - Horizon file list. The horizon files must contain xyz coordinates or
                     inline number, cross-line number and z coordinates, or both.
    :param hor_sep: (String) - Default is '\t'. The column delimiter in horizon files.
    :param hor_header: (Integer or list of integers) - Default is None, which means no header in horizon file as column
                       names. Row number(s) to use as column names.
    :param hor_col_names: (List of strings) - Default is None, which means to use the header in horizon file as column
                          names. The column names of horizon data. If not None, the hor_col_names will be used as column
                          names of the horizon data, and if there are headers in horizon files, they will be replaced by
                          hor_col_names.
    :param hor_x: (String) - X coordinate name of horizon data.
    :param hor_y: (String) - Y coordinate name of horizon data.
    :param hor_il: (String) - Inline number name of horizon data.
    :param hor_xl: (String) - Cross-line number name of horizon data.
    :param hor_z: (String) - Z coordinate name of horizon data.
    :param hor_xwin: (Float) - Default is 25.0 when on=='xy' and 1.0 when on=='ix'.
                     The window in which the horizon x coordinates (or inline numbers) will be matched with the
                     cube x coordinates (or inline numbers).
    :param hor_ywin: (Float) - Default is 25.0 when on=='xy' and 1.0 when on=='ix'.
                     The window in which the horizon x coordinates (or cross-line numbers) will be matched with the
                     cube y coordinates (or cross-line numbers).
    :param hor_zwin: (Float) - Default is 2.0. The window in which the horizon z coordinates will be matched with the
                     cube z coordinates.
    :param show_slice: (Bool) - Default is True. If True, will show the cube data as orthogonal slices. Otherwise will
                       show the cube's surface.
    :param show_axes: (Bool) - Default is True. Whether to show the coordinate axes.
    """
    # Load cube data from SEG-Y file.
    if cube_file is not None:
        with segyio.open(cube_file) as f:
            f.mmap()  # Memory mapping for faster reading.
            inline = f.ilines  # Get inline numbers.
            xline = f.xlines  # Get cross-line numbers.
            x = np.zeros(len(inline), dtype='float32')
            y = np.zeros(len(xline), dtype='float32')
            for i in range(len(inline)):
                x[i] = f.header[i * len(xline)][header_x] * scl_x  # Get trace x coordinates.
            for i in range(len(xline)):
                y[i] = f.header[i][header_y] * scl_y  # Get trace y coordinates.
            z = f.samples  # Get z coordinate of every trace.
            # Print cube info.
            print('Cube info:')
            print('Inline: %d-%d [%d lines]' % (inline[0], inline[-1], len(inline)))
            print('Xline: %d-%d [%d lines]' % (xline[0], xline[-1], len(xline)))
            print('X Range: [%d-%d] [%d samples]' % (x[0], x[-1], len(x)))
            print('Y Range: [%d-%d] [%d samples]' % (y[0], y[-1], len(y)))
            print('Z Range: [%d-%d] [%d samples]' % (z[0], z[-1], len(z)))
            data = segyio.tools.cube(f)  # Load cube data.
        f.close()
    # Directly input cube data.
    elif cube_data is not None:
        data = cube_data.copy()
        if data.ndim != 3:
            raise ValueError("The input data have %d dimension(s) instead of 3" % data.ndim)
        cube_shape = np.shape(cube_data)
        print('Cube info:')
        print('Cube shape:', cube_shape)
        x = np.arange(0, cube_shape[0], 1)
        y = np.arange(0, cube_shape[1], 1)
        z = np.arange(0, cube_shape[2], 1)
        print('X Range: [%d-%d] [%d samples]' % (x[0], x[-1], len(x)))
        print('Y Range: [%d-%d] [%d samples]' % (y[0], y[-1], len(y)))
        print('Z Range: [%d-%d] [%d samples]' % (z[0], z[-1], len(z)))
        on = 'xy'
    # Initiate plotter.
    flag = 0
    if p is None:
        pv.set_plot_theme('ParaView')
        p = BackgroundPlotter(lighting='three lights')
        flag += 1
    sargs = dict(height=0.5, vertical=True, position_x=0.85, position_y=0.2,
                 label_font_size=14, title_font_size=18)  # The scalar bar arguments.
    # Plot the cube.
    if on == 'xy':  # Use trace coordinates as x, y coordinates.
        x_cube, y_cube, z_cube = np.meshgrid(x, y, z, indexing='ij')
    elif on == 'ix':  # Use inline and cross-line numbers as x, y coordinates.
        x_cube, y_cube, z_cube = np.meshgrid(inline, xline, z, indexing='ij')
    else:
        raise ValueError("The parameter 'on' can only be 'xy' or 'ix'.")
    cube_grid = pv.StructuredGrid(x_cube, y_cube, z_cube)  # Create structured grid for cube.
    if scale is not None:
        cube_grid.scale(scale)  # Scale the grid size.
    cube_grid[value_name] = np.ravel(data, order='F')  # Map cube data to the grid.
    if show_slice:  # Show interactive orthogonal slices.
        p.add_mesh_slice_orthogonal(cube_grid, cmap=colormap, scalar_bar_args=sargs)
    else:  # Show the cube surface.
        p.add_mesh(mesh=cube_grid, scalars=value_name, cmap=colormap, scalar_bar_args=sargs)
    # Plot horizons.
    if hor_list is not None:
        for hor in hor_list:
            print("Loading horizon from file '%s'" % hor)
            # Load horizons from files.
            if hor_col_names is None:
                df_hor = pd.read_csv(hor, delimiter=hor_sep)
            else:
                df_hor = pd.read_csv(hor, delimiter=hor_sep, header=hor_header, names=hor_col_names)
            # Get horizon info.
            if on == 'xy':
                if df_hor[[hor_x, hor_y, hor_z]].isna().any().any():
                    print('Can not process horizon with missing values in coordinates.')
                    continue
                x_temp = df_hor[hor_x].drop_duplicates().values
                y_temp = df_hor[hor_y].drop_duplicates().values
                print('Horizon info:')
                print('X Range: %d-%d [%d samples]' % (x_temp[0], x_temp[-1], len(x_temp)))
                print('Y Range: %d-%d [%d samples]' % (y_temp[0], y_temp[-1], len(y_temp)))
            elif on == 'ix':
                if df_hor[[hor_il, hor_xl, hor_z]].isna().any().any():
                    print('Can not process horizon with missing values in coordinates.')
                    continue
                x_temp = df_hor[hor_il].drop_duplicates().values
                y_temp = df_hor[hor_xl].drop_duplicates().values
                print('Horizon info:')
                print('Inline Range: %d-%d [%d samples]' % (x_temp[0], x_temp[-1], len(x_temp)))
                print('Xline Range: %d-%d [%d samples]' % (y_temp[0], y_temp[-1], len(y_temp)))
            if len(df_hor) != len(x_temp) * len(y_temp):
                print("The horizon with length %d can not match inferred grid size (%d, %d)." %
                      (len(df_hor), len(x_temp), len(y_temp)))
                continue
            # Get data from cube to horizon.
            if on == 'xy':
                x_dist = cdist(np.reshape(df_hor[hor_x].values, (-1, 1)), np.reshape(x, (-1, 1)),
                               metric='minkowski', p=1)
                y_dist = cdist(np.reshape(df_hor[hor_y].values, (-1, 1)), np.reshape(y, (-1, 1)),
                               metric='minkowski', p=1)
            elif on == 'ix':
                x_dist = cdist(np.reshape(df_hor[hor_il].values, (-1, 1)), np.reshape(inline, (-1, 1)),
                               metric='minkowski', p=1)
                y_dist = cdist(np.reshape(df_hor[hor_xl].values, (-1, 1)), np.reshape(xline, (-1, 1)),
                               metric='minkowski', p=1)
            z_dist = cdist(np.reshape(df_hor[hor_z].values, (-1, 1)), np.reshape(z, (-1, 1)),
                           metric='minkowski', p=1)
            indx = np.argmin(x_dist, axis=1)
            indy = np.argmin(y_dist, axis=1)
            indz = np.argmin(z_dist, axis=1)
            if hor_xwin is None:
                if on == 'xy':
                    hor_xwin = 25.0
                elif on == 'ix':
                    hor_xwin = 1.0
            if hor_ywin is None:
                if on == 'xy':
                    hor_ywin = 25.0
                elif on == 'ix':
                    hor_ywin = 1.0
            x_dist_min = np.amin(x_dist, axis=1)
            y_dist_min = np.amin(y_dist, axis=1)
            z_dist_min = np.amin(z_dist, axis=1)
            ix = np.squeeze(np.argwhere(x_dist_min < hor_xwin))
            iy = np.squeeze(np.argwhere(y_dist_min < hor_ywin))
            iz = np.squeeze(np.argwhere(z_dist_min < hor_zwin))
            ind = np.intersect1d(np.intersect1d(ix, iy), iz)
            df_hor.loc[ind, value_name] = data[indx[ind], indy[ind], indz[ind]]
            if on == 'xy':
                x_hor = np.reshape(df_hor[hor_x].values, (len(x_temp), len(y_temp)), order='C')
                y_hor = np.reshape(df_hor[hor_y].values, (len(x_temp), len(y_temp)), order='C')
            elif on == 'ix':
                x_hor = np.reshape(df_hor[hor_il].values, (len(x_temp), len(y_temp)), order='C')
                y_hor = np.reshape(df_hor[hor_xl].values, (len(x_temp), len(y_temp)), order='C')
            z_hor = np.reshape(df_hor[hor_z].values, (len(x_temp), len(y_temp)), order='C')
            v_hor = np.reshape(df_hor[value_name].values, (len(x_temp), len(y_temp)), order='C')
            # Create structured grid for horizon.
            hor_grid = pv.StructuredGrid(x_hor, y_hor, z_hor)
            if scale is not None:
                hor_grid.scale(scale)  # Scale the grid.
            hor_grid[value_name] = np.ravel(v_hor, order='F')  # Map horizon data to the grid.
            p.add_mesh(hor_grid, scalars=value_name, cmap=colormap)
    if fig_name is not None:
        p.add_text(fig_name, font_size=18)
    if on == 'ix':
        p.add_axes(xlabel='Inline', ylabel='Xline')  # Add an interactive axes widget in the bottom left corner.
    else:
        p.add_axes()
    if show_axes:  # Show coordinate axes.
        if on == 'xy':
            p.show_bounds(xlabel='X', ylabel='Y', zlabel='Z')
        elif on == 'ix':
            p.show_bounds(xlabel='Inline', ylabel='Xline', zlabel='Z')
    if flag == 1:
        p.app.exec_()  # Show all figures.

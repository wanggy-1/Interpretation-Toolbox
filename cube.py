import numpy as np
import pandas as pd
import segyio
import time
import sys
import os
from well_log import resample_log
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist


def FSDI(seismic_file, log_dir, output_file, vertical_well=True, abnormal_value=-999, method='average', scale=True,
         log_name=None, depth_name=None, coord_name=None, seis_name=None, well_name=None, weight=None,
         well_location_file=None, well_name_loc=None, coord_name_loc=None):
    """
    Feature and distance based interpolation (FSDI) for cube.
    :param seismic_file: (Strings or list of strings) - Seismic attributes file name (segy or sgy format).
                         For single file, directly enter file name.
                         For multiple files, enter file names as list of strings, e.g. ['a.sgy', 'b.sgy'].
    :param log_dir: (String) - Well log file directory.
    :param output_file: (String or list of strings) - Output file name (ASCII) for interpolation results.
                        For single file, directly enter file name.
                        For multiple files, enter file names as list of strings, e.g. ['a.txt', 'b.txt'].
                        Note that the number of output files should match with the number of logs.
    :param vertical_well: (Bool) - Whether the wells are vertical.
                          If True, will process wells as vertical wells, the well log files must have two columns:
                          depth (depth_name) and log (log_name).
                          If False, will process wells as inclined wells, the well log files must
                          have four columns: x (coord_name[0]), y (coord_name[1]), depth (depth_name) and log (log_name).
    :param abnormal_value: (Float) - Default is -999. The abnormal value in log column.
    :param method: (Strings) - Default is 'average'. Well log re-sampling method.
                               Options: 'nearest', 'average', 'median', 'rms', 'most_frequent'.
    :param scale: (Bool) - Default is True. Whether to scale coordinates and seismic attributes to 0 and 1 with
                           MinMaxScalar.
    :param log_name: (String or list of strings) - Well log name.
                     For single log, directly enter log name like 'porosity'.
                     For multiple logs, enter log names as list of strings, e.g. ['Litho_Code', 'porosity'].
    :param depth_name: (String) - Depth column name.
    :param coord_name: (List of strings) - Well coordinate column names, e.g. ['X', 'Y'].
    :param seis_name: (String or list of strings) - Seismic attribute column name.
                      For single attribute, directly enter attribute name like 'amplitude'.
                      For multiple attributes, enter attribute names as list of strings, e.g. ['amplitude', 'phase'].
    :param well_name: (String) - Well name column name.
    :param well_location_file: (String) - Well location file name.
    :param well_name_loc: (String) - Well name column name in well location file. Only used when vertical_well=True.
    :param coord_name_loc: (List of strings) - Well coordinate column nae in well location file.
                           Only used when vertical_well=True.
    :return: cube_itp: (numpy.ndarray) - A 4d array contains the interpolation results.
                       cube_itp[inline, xline, samples, interp_logs].
    """
    # Read seismic file.
    t1 = time.perf_counter()  # Timer.
    # If multiple files.
    if isinstance(seismic_file, str):
        Nfile = 1
        seismic_file = [seismic_file]
    elif isinstance(seismic_file, list):
        Nfile = len(seismic_file)
    else:
        raise ValueError('Seismic file must be string or list of strings.')
    # Initiate list to store multiple files.
    seis = []
    for n in range(Nfile):
        with segyio.open(seismic_file[n]) as f:
            print('Read seismic data from file: ', seismic_file[n])
            # Memory map file for faster reading (especially if file is big...)
            mapped = f.mmap()
            if mapped:
                print('\tSeismic file is memory mapped.')
            # Print file information.
            print('\tFile info:')
            print('\tinline range: %d-%d [%d lines]' % (f.ilines[0], f.ilines[-1], len(f.ilines)))
            print('\tcrossline range: %d-%d [%d lines]' % (f.xlines[0], f.xlines[-1], len(f.xlines)))
            print('\tTime range: %dms-%dms [%d samples]' % (f.samples[0], f.samples[-1], len(f.samples)))
            dt = segyio.tools.dt(f) / 1000
            print('\tSampling interval: %.1fms' % dt)
            print('\tTotal traces: %d' % f.tracecount)
            # Read seismic data.
            cube = segyio.tools.cube(f)
            # Read sampling time.
            t = f.samples
            # Extract trace coordinates from trace header.
            x = np.zeros(shape=(f.tracecount, ), dtype='float32')
            y = np.zeros(shape=(f.tracecount, ), dtype='float32')
            for i in range(f.tracecount):
                sys.stdout.write('\rExtracting trace coordinates: %.2f%%' % ((i+1) / f.tracecount * 100))
                x[i] = f.header[i][73] * 1e-1  # Adjust coordinate according to actual condition.
                y[i] = f.header[i][77] * 1e-1  # Adjust coordinate according to actual condition.
            sys.stdout.write('\n')
            # Re-shape the trace coordinates array to match the seismic data cube.
            x = x.reshape([len(f.ilines), len(f.xlines)], order='C')
            y = y.reshape([len(f.ilines), len(f.xlines)], order='C')
        # print('x:\n', x)
        # print('y:\n', y)
        f.close()
        seis.append(cube)
    # Read well log file.
    log_list = os.listdir(log_dir)  # Well log file list.
    # Initiate control points data frame.
    df_ctp = pd.DataFrame()
    if vertical_well:
        # Read well locations.
        df_loc = pd.read_csv(well_location_file, delimiter='\s+')
    cnt = 0
    for log_file in log_list:
        df = pd.read_csv(os.path.join(log_dir, log_file), delimiter='\t')
        # Drop rows with abnormal value.
        drop_col = [x for x in range(len(df)) if df.loc[x, log_name] == abnormal_value]
        df.drop(index=drop_col, inplace=True)
        df.reset_index(drop=True, inplace=True)
        # Re-sample well log by seismic sampling interval.
        df = resample_log(df, delta=dt, depth_col=depth_name, log_col=log_name, method=method)
        if vertical_well:
            # Extract well coordinates from well (only for vertical wells).
            well_coord = df_loc.loc[df_loc[well_name_loc] == log_file[:-4], coord_name_loc].values
            if (np.squeeze(well_coord)[0] > np.amax(x) or np.squeeze(well_coord)[0] < np.amin(x)) and \
                    (np.squeeze(well_coord)[1] > np.amax(y) or np.squeeze(well_coord)[1] < np.amin(y)):
                continue  # Check if this well is in target area.
        # Change well coordinates to their nearest seismic data coordinates.
        seis_coord = np.c_[x.ravel(), y.ravel()]  # Seismic data coordinates.
        if vertical_well:  # For vertical well.
            ind = np.argmin(np.sqrt(np.sum((well_coord - seis_coord) ** 2, axis=1)))
            well_coord = seis_coord[ind]
            well_coord = np.ones(shape=[len(df), 2]) * well_coord
        else:  # For inclined well.
            for i in range(len(df)):
                log_coord = df.loc[i, coord_name]
                if (np.squeeze(log_coord)[0] > np.amax(x) or np.squeeze(log_coord)[0] < np.amin(x)) and \
                        (np.squeeze(log_coord)[1] > np.amax(y) or np.squeeze(log_coord)[1] < np.amin(y)):
                    continue  # Check if this log location is in target area.
                ind = np.argmin(np.sqrt(np.sum((log_coord - seis_coord) ** 2, axis=1)))
                log_coord = seis_coord[ind]
                df.loc[i, coord_name] = log_coord
        if vertical_well:
            # Add well coordinate to data frame.
            data = np.c_[well_coord, df.values]
            if isinstance(log_name, str):
                df = pd.DataFrame(data=data, columns=coord_name + [depth_name, log_name])
            if isinstance(log_name, list):
                df = pd.DataFrame(data=data, columns=coord_name + [depth_name] + log_name)
        # Add seismic features at control points (well log) to data frame.
        seis_ctp = np.zeros(shape=[len(df), Nfile], dtype='float32')
        if vertical_well:  # Vertical well.
            indx, indy = np.squeeze(np.argwhere((x == well_coord[0, 0]) & (y == well_coord[0, 1])))
            indz0 = np.squeeze(np.argwhere(t == df[depth_name].min()))
            indz1 = np.squeeze(np.argwhere(t == df[depth_name].max()))
            for i in range(Nfile):
                seis_ctp[:, i] = seis[i][indx, indy, indz0:indz1+1]
        else:  # Inclined well.
            for i in range(len(df)):
                indx, indy = np.squeeze(np.argwhere((x == df.loc[i, coord_name[0]]) & (y == df.loc[i, coord_name[1]])))
                indz = np.squeeze(np.argwhere(t == df.loc[i, depth_name]))
                for j in range(Nfile):
                    seis_ctp[i, j] = seis[j][indx, indy, indz]
        df[seis_name] = seis_ctp
        df[well_name] = log_file[:-4]
        df_ctp = df_ctp.append(df, ignore_index=True)
        cnt += 1
        sys.stdout.write('\rAssembling well logs: %.2f%%' % (cnt / len(log_list) * 100))
    sys.stdout.write('\n')
    print(df_ctp)
    # FSDInterpolation.
    # Scale.
    if scale:
        # MinMaxScalar.
        xy_scalar = MinMaxScaler()  # x and y scalar.
        t_scalar = MinMaxScaler()  # Two-way time scalar.
        seis_scalar = MinMaxScaler()  # Seismic attributes scalar.
        # Fit.
        xy_scalar.fit(np.c_[x.ravel(order='C'), y.ravel(order='C')])
        t_scalar.fit(t.reshape(-1, 1))
        seis_r = np.zeros(shape=[len(seis[0].ravel(order='C')), Nfile], dtype='float32')
        for i in range(Nfile):
            seis_r[:, i] = seis[i].ravel(order='C')
        seis_scalar.fit(seis_r)
    # Determine interpolation depth range by well log depth range.
    depth_min = df_ctp[depth_name].min()
    depth_max = df_ctp[depth_name].max()
    # Initiate interpolation 3d array (this will be the result).
    ind_min = np.squeeze(np.argwhere(t == depth_min))
    ind_max = np.squeeze(np.argwhere(t == depth_max))
    t_cut = t[ind_min: ind_max+1]
    np.savetxt('depth.txt', t_cut.reshape(-1, 1))
    if isinstance(log_name, list):
        cube_itp = np.zeros(shape=[seis[0].shape[0], seis[0].shape[1], len(t_cut), len(log_name)], dtype='float32')
    elif isinstance(log_name, str):
        cube_itp = np.zeros(shape=[seis[0].shape[0], seis[0].shape[1], len(t_cut), 1], dtype='float32')
    else:
        raise ValueError('Log column name can only be string or list of strings.')
    # Get control points.
    if isinstance(seis_name, str):
        seis_name = [seis_name]
    ctp = df_ctp[coord_name + [depth_name] + seis_name].values
    if scale:
        ctp[:, 0:2] = xy_scalar.transform(ctp[:, 0:2])  # Scale control points' x and y coordinates.
        ctp[:, 2] = np.squeeze(t_scalar.transform(ctp[:, 2].reshape(-1, 1)))  # Scale control points' TWT coordinates.
        if ctp[:, 3:].ndim == 1:
            ctp[:, 3:] = np.squeeze(seis_scalar.transform(ctp[:, 3:].reshape(-1, 1)))
        else:
            ctp[:, 3:] = seis_scalar.transform(ctp[:, 3:])  # Scale control points' seismic attributes.
    # Compute distance map trace by trace.
    for i in range(cube_itp.shape[0]):  # Inlines.
        for j in range(cube_itp.shape[1]):  # Trace number.
            sys.stdout.write('\rInterpolating trace %d/%d [%.2f%%]' %
                             (i*cube_itp.shape[1]+j+1, cube_itp.shape[0]*cube_itp.shape[1],
                              (i*cube_itp.shape[1]+j+1) / (cube_itp.shape[0]*cube_itp.shape[1]) * 100))
            x_itp = x[i, j] * np.ones(cube_itp.shape[2], dtype='float32')  # x-coordinate.
            y_itp = y[i, j] * np.ones(cube_itp.shape[2], dtype='float32')  # y-coordinate.
            t_itp = t_cut.copy()  # Two-way time.
            seis_itp = np.zeros(shape=[cube_itp.shape[2], Nfile], dtype='float32')  # Seismic attributes.
            for n in range(Nfile):
                seis_itp[:, n] = seis[n][i, j, ind_min: ind_max+1]
            itp = np.c_[x_itp, y_itp, t_itp, seis_itp]  # Points to interpolate.
            if scale:
                itp[:, 0:2] = xy_scalar.transform(itp[:, 0:2])  # Scale interpolate points' x and y coordinates.
                itp[:, 2] = np.squeeze(t_scalar.transform(itp[:, 2].reshape(-1, 1)))  # Scale interpolate points' TWT.
                if itp[:, 3:].ndim == 1:
                    itp[:, 3:] = np.squeeze(seis_scalar.transform(itp[:, 3:].reshape(-1, 1)))
                else:
                    itp[:, 3:] = seis_scalar.transform(itp[:, 3:])  # Scale interpolate points' seismic attributes.
            dist_map = cdist(itp, ctp, metric='euclidean')  # Compute distance map.
            # Get column index of minimum distance.
            min_idx = np.argmin(dist_map, axis=1)
            # Interpolate log values according to minimum distance.
            log_itp = df_ctp.loc[min_idx, log_name].values
            if log_itp.ndim == 1:
                log_itp = log_itp.reshape(-1, 1)
            cube_itp[i, j, :, :] = log_itp  # Write trace to cube.
    sys.stdout.write('\n')
    # Save interpolation result.
    x_out = x.ravel(order='C')
    y_out = y.ravel(order='C')
    if isinstance(output_file, str):
        output_file = [output_file]
    for i in range(cube_itp.shape[3]):
        sys.stdout.write('Writing interpolation result to file %s...' % output_file[i])
        cube_out = cube_itp[:, :, :, i].reshape([cube_itp.shape[0] * cube_itp.shape[1], cube_itp.shape[2]], order='C')
        np.savetxt(output_file[i], np.c_[x_out, y_out, cube_out], delimiter='\t')
        sys.stdout.write('Done.\n')
    t2 = time.perf_counter()
    print('Process time: %.2fs' % (t2 - t1))
    return cube_itp


if __name__ == '__main__':
    seismic_file = ['/nfs/opendtect-data/Niuzhuang/Export/seismic_east.sgy',
                    '/nfs/opendtect-data/Niuzhuang/Export/vpvs_east.sgy',
                    '/nfs/opendtect-data/Niuzhuang/Export/sp_east.sgy']
    log_dir = '/nfs/opendtect-data/Niuzhuang/Well logs/LithoCodeForPetrel-time'  # Well log directory.
    well_location_file = '/nfs/opendtect-data/Niuzhuang/Well logs/well_locations.prn'  # Well location file.
    log_name = 'Litho_Code'
    depth_name = 'TWT'
    coord_name = ['X', 'Y']
    seis_name = ['SeisAmp', 'VpVs', 'SP']
    well_name = 'WellName'
    output_file = 'Litho_Code_2.txt'
    well_name_loc = 'well_name'
    coord_name_loc = ['well_X', 'well_Y']
    result = FSDI(seismic_file=seismic_file, log_dir=log_dir, output_file=output_file, method='most_frequent',
                  log_name=log_name, depth_name=depth_name, coord_name=coord_name, seis_name=seis_name, well_name=well_name,
                  well_location_file=well_location_file, well_name_loc=well_name_loc, coord_name_loc=coord_name_loc)
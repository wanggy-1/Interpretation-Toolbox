##
# Demonstration of interpolating segmented well log data.
from well_log import *

# The segmented well log file name.
log_file = '/nfs/opendtect-data/Niuzhuang/Well logs/LithoCodeForPetrel/DK1.txt'
# Read the segmented well log as data frame.
df_log = pd.read_csv(log_file, delimiter='\s+')
# Print and check the segmented well log data frame.
print('Segmented well log data frame:\n', df_log)
# Interpolate the segmented well log data.
sampling_interval = 0.125  # Well log sampling interval (0.125m), this is also the interpolation step.
log_col_name = 'Litho_Code'  # Well log column name in df_log.
depth_col_name = 'Depth'  # Depth column name in df_log
top_col_name = 'TopDepth'  # Top boundary column name in df_log.
bottom_col_name = 'BottomDepth'  # Bottom boundary column name in df_log.
nominal = True  # Lithology well log data are nominal data.
df_log_interp = log_interp(df=df_log, step=sampling_interval, log_col=log_col_name, depth_col=depth_col_name,
                           top_col=top_col_name, bottom_col=bottom_col_name, nominal=nominal)
# Print interpolated well log data frame.
print('Interpolated well log data frame:\n', df_log_interp)


##
# Demonstration of converting well logs from depth-domain to time-domain (single well log).
from well_log import *

# Depth-domain well log file name.
log_file = '/nfs/opendtect-data/Niuzhuang/Well logs/LithoCodeForPetrel-interpolated/DK1_interpolated.txt'
# Depth-time relation file name.
dt_file = '/nfs/opendtect-data/Niuzhuang/Well logs TD (new)/DK1.txt'
# Read depth-domain well log as data frame.
df_log = pd.read_csv(log_file, delimiter='\t')
# Read depth-time relation as data frame.
df_dt = pd.read_csv(dt_file, delimiter='\t')
# Print and check depth-domain well log data frame.
print('Depth-domain well log data frame:\n', df_log)
# Print and check depth-time relation data frame.
print('Depth-time relation data frame:\n', df_dt)
# Convert depth-domain well log to time-domain well log.
time_col_name = 'TWT'  # Time column name in df_dt.
depth_col_name = 'Depth'  # Depth column name in df_log and df_dt (Two files, same depth column name).
log_col_name = 'Litho_Code'  # Well log column name in df_log.
fill_nan = -999  # There may be NaN after conversion. Fill it with this value (-999).
nominal = True  # Lithology well log data are nominal data.
df_tlog = time_log(df_dt=df_dt, df_log=df_log, time_col=time_col_name, log_depth_col=depth_col_name,
                   dt_depth_col=depth_col_name, log_col=log_col_name, fill_nan=fill_nan, nominal=nominal)
# Print time-domain well log data frame.
print('Time domain well log data frame:\n', df_tlog)


##
# Demonstration of converting well logs from depth-domain to time-domain (multiple well logs).
from well_log import *

# Depth-domain well log file name.
log_file = '/nfs/opendtect-data/Niuzhuang/Well logs/Por-Perm-Sw/DK1-Por-Perm-Sw.txt'
# Depth-time relation file name.
dt_file = '/nfs/opendtect-data/Niuzhuang/Well logs TD (new)/DK1.txt'
# Read depth-domain well log as data frame.
df_log = pd.read_csv(log_file, delimiter='\s+')
# Read depth-time relation as data frame.
df_dt = pd.read_csv(dt_file, delimiter='\t')
# Print and check depth-domain well log data frame.
print('Depth-domain well log data frame:\n', df_log)
# Print and check depth-time relation data frame.
print('Depth-time relation data frame:\n', df_dt)
# Convert depth-domain well log to time-domain well log.
time_col_name = 'TWT'  # Time column name in df_dt.
log_depth_col_name = 'DEPTH'  # Depth column name in df_log.
dt_depth_col_name = 'Depth'  # Depth column name in df_dt.
log_col_name = ['POR', 'PERM', 'SW']  # Well log column name in df_log.
fill_nan = -999  # There may be NaN after conversion. Fill it with this value (-999).
nominal = False  # Porosity, permeability and water-saturation data are numeric data.
df_tlog = time_log(df_dt=df_dt, df_log=df_log, time_col=time_col_name, log_depth_col=log_depth_col_name,
                   dt_depth_col=dt_depth_col_name, log_col=log_col_name, fill_nan=fill_nan, nominal=nominal)
# Print time-domain well log data frame.
print('Time domain well log data frame:\n', df_tlog)


##
# Demonstration of resampling time-domain well logs by seismic sampling interval (single well log).
from well_log import *

# Time-domain Well log file name.
log_file = '/nfs/opendtect-data/Niuzhuang/Well logs/LithoCodeForPetrel-time/DK1.txt'
# Read time-domain well log as data frame.
df_log = pd.read_csv(log_file, delimiter='\t')
# Print and check the well log, especially column names, which will be used in the following function.
print('Original well log data frame (5000~5099 rows):\n', df_log[5000:5100])
# Re-sample well log by seismic sampling interval (2ms)
sampling_interval = 2  # 2ms sampling interval.
depth_col_name = 'TWT'  # Depth column name in df_log.
log_col_name = 'Litho_Code'  # Well log column name in df_log.
resample_method = 'nearest'  # Choose re-sampling method. For nominal data, 'nearest' and 'most_frequent' are suitable.
abnormal_value = -999  # There may be some abnormal value in original well log data, delete them.
nominal = True  # Lithology well log data are nominal data.
df_log_res = resample_log(df_log=df_log, delta=sampling_interval, depth_col=depth_col_name, log_col=log_col_name,
                          method=resample_method, abnormal_value=abnormal_value, nominal=nominal)
# Print re-sampled well log data frame.
print('Re-sampled well log data frame (200~299 rows):\n', df_log_res[200:300])


##
# Demonstration of resampling time-domain well logs by seismic sampling interval (multiple well logs).
from well_log import *

# Time-domain Well log file name.
log_file = '/nfs/opendtect-data/Niuzhuang/Well logs/Por-Perm-Sw-time/DK1.txt'
# Read time-domain well log as data frame.
df_log = pd.read_csv(log_file, delimiter='\t')
# Print and check the well log, especially column names, which will be used in the following function.
print('Original well log data frame (5000~5099 rows):\n', df_log[5000:5100])
# Re-sample well log by seismic sampling interval (2ms)
sampling_interval = 2  # 2ms sampling interval.
depth_col_name = 'TWT'  # Depth column name in df_log.
log_col_name = ['POR', 'PERM', 'SW']  # Well log column names in df_log.
resample_method = 'average'  # Choose re-sampling method. For numeric data, 'average', 'median' and 'rms' are suitable.
abnormal_value = -999  # There may be some abnormal value in original well log data, delete them.
nominal = False  # Porosity, permeability and water-saturation are numeric data.
df_log_res = resample_log(df_log=df_log, delta=sampling_interval, depth_col=depth_col_name, log_col=log_col_name,
                          method=resample_method, abnormal_value=abnormal_value, nominal=nominal)
# Print re-sampled well log data frame.
print('Re-sampled well log data frame (210~299 rows):\n', df_log_res[210:300])


##
# Demonstration of making 2D cross-plot with well logging data.
from well_log import *

# Load well logging data.
file = '/nfs/opendtect-data/Niuzhuang/Well logs/W584.csv'
df = pd.read_csv(file)
# Choose well logs as x and y coordinates of scatters.
x = 'AC'  # The column name in df.
y = 'DEN'  # The column name in df.
# Choose well log as colors of scatters.
c = 'GR'  # The column name in df.
# Choose a colormap.
cmap = 'rainbow'
# Define x and y axis name.
x_name = 'Acoustic Compressional - us/m'
y_name = 'Bulk Density - g/cm3'
# Define color bar name.
cb_name = 'Gamma Ray - API'
# Make cross-plot.
cross_plot2D(df=df, x=x, y=y, c=c, cmap=cmap, xlabel=x_name, ylabel=y_name, title='W584', colorbar=cb_name)


##
# Demonstration of making 3D cross-plot of well logging data.
from well_log import *

# Load well logging data.
file = '/nfs/opendtect-data/Niuzhuang/Well logs/W584.csv'
df = pd.read_csv(file)
# Choose well logs as x, y and z coordinates of scatters.
x = 'AC'  # The column name in df.
y = 'DEN'  # The column name in df.
z = 'SP'  # The column name in df.
# Choose well log as colors of scatters.
c = 'GR'  # The column name in df.
# Choose a color map.
cmap = 'rainbow'
# Define x, y and z axis names.
x_name = 'Acoustic Compressional - us/m'
y_name = 'Bulk Density - g/cm3'
z_name = 'Spontaneous Potential - mV'
# Define color bar name.
cb_name = 'Gamma Ray - API'
cross_plot3D(df=df, x=x, y=y, z=z, c=c, cmap=cmap, xlabel=x_name, ylabel=y_name, zlabel=z_name, colorbar=cb_name,
             title='W584')


##
# Demonstration of visualizing well log.
from well_log import *

# Load well logging data.
file = '/nfs/opendtect-data/Niuzhuang/Well logs/W584.csv'
df = pd.read_csv(file)
# Depth column name.
depth = 'DEPTH'
# Log column name.
log = 'POR'
# Choose a color map.
cmap = 'rainbow'
# Set depth range to visualize.
ylim = [2400, 2600]
# Define x axis name.
x_name = 'Porosity - %'
plotlog(df, depth=depth, log=log, cmap=cmap, ylim=ylim, xlabel='Porosity - %', title='W584', fill_log=True)


##
# This is a demonstration of using FSDI to interpolate lithology on horizons.
from horizon import *
from matplotlib.colors import LinearSegmentedColormap

# Inputs.
hor_list = ['z1', 'z2', 'z3', 'z4', 'z5', 'z6']  # Horizons on which to interpolate.
hor_x, hor_y, hor_z = 'X', 'Y', 'Z'  # 3D coordinates names of horizons.
well_x, well_y, well_z = 'well_X', 'well_Y', 'TWT'  # Well log z coordinate name.
log_abnormal = -999  # Abnormal value in well log.
feature_list = ['gailv_sp', 'seismic']  # Features which are used to control the interpolation.
weight = [1, 1, 1, 1, 1]  # Weight of [x, y, z, feature1, feature2].
target = 'Litho_Code'  # Interpolation target.
hor_folder = '/nfs/opendtect-data/Niuzhuang/Export'  # Folder which contains horizon data.
log_folder = '/nfs/opendtect-data/Niuzhuang/Well logs/LithoCodeForPetrel-time'  # Folder which contains well log data.
well_xy_file = '/nfs/opendtect-data/Niuzhuang/Well logs/well_locations.prn'  # Well location file name.
hor_suffix = '-features-clustering.dat'  # Horizon file name suffix.
marker_color = ['grey', 'limegreen', 'cyan', 'gold', 'darkviolet']  # Color of 5 lithology classes.
labels = ['mudstone', 'lime-mudstone', 'siltstone', 'sandstone', 'gravel-sandstone']  # Labels of 5 lithology classes.
class_code = [0, 1, 2, 3, 4]  # Lithology codes.

# Load well location.
df_well_xy = pd.read_csv(well_xy_file, delimiter='\s+')

# Interpolation and visualization.
for hor in hor_list:
    sys.stdout.write('Interpolating %s on %s...' % (target, hor))
    df_hor = pd.read_csv(os.path.join(hor_folder, hor + hor_suffix), delimiter='\t',
                         usecols=[hor_x, hor_y, hor_z] + feature_list)
    control = horizon_log(df_horizon=df_hor[[hor_x, hor_y, hor_z]], df_well_coord=df_well_xy, log_file_path=log_folder,
                          sep='\t', well_x_col=well_x, well_y_col=well_y, log_t_col=well_z,
                          horizon_x_col=hor_x, horizon_y_col=hor_y, horizon_t_col=hor_z,
                          log_value_col=target, log_abnormal_value=log_abnormal)
    df_itp, _ = FSDI_horizon(df_horizon=df_hor, df_control=control, coord_col=[hor_x, hor_y, hor_z],
                             feature_col=feature_list, log_col=target, weight=weight)
    sys.stdout.write(' Done.\n')
    cm = LinearSegmentedColormap.from_list('custom', marker_color, len(marker_color))  # Customized colormap.
    visualize_horizon(df=df_itp, x_name=hor_x, y_name=hor_y, value_name=target, cmap=cm, vmin=min(class_code),
                      vmax=max(class_code), nominal=True, class_code=class_code, class_label=labels,
                      fig_name=f'{hor}-{target}')
    plot_markers(df=control, x_col=hor_x, y_col=hor_y, class_col=target, wellname_col='WellName',
                 class_code=class_code, class_label=labels, colors=marker_color)


##
# Demonstration of using FSDI to interpolate lithology between horizons.
from horizon import *

seis_file = '/nfs/opendtect-data/Niuzhuang/Export/seismic_east.sgy'
hor_file = ['/nfs/opendtect-data/Niuzhuang/Export/T4_east_dense.dat',
            '/nfs/opendtect-data/Niuzhuang/Export/z1_east_dense.dat',
            '/nfs/opendtect-data/Niuzhuang/Export/z2_east_dense.dat',
            '/nfs/opendtect-data/Niuzhuang/Export/z3_east_dense.dat',
            '/nfs/opendtect-data/Niuzhuang/Export/z4_east_dense.dat',
            '/nfs/opendtect-data/Niuzhuang/Export/z5_east_dense.dat',
            '/nfs/opendtect-data/Niuzhuang/Export/z6_east_dense.dat',
            '/nfs/opendtect-data/Niuzhuang/Export/T6_east_dense.dat']
log_folder = '/nfs/opendtect-data/Niuzhuang/Well logs/LithoCodeForPetrel-time'
well_loc_file = '/nfs/opendtect-data/Niuzhuang/Well logs/well_locations.prn'
output_file = '/nfs/opendtect-data/Niuzhuang/Litho_Code_11.txt'
FSDI_interhorizon(seis_file=seis_file, seis_name='seis_amp',
                  horizon_file=hor_file, horizon_col=['x', 'y', 'inline', 'xline', 't'],
                  log_dir=log_folder, log_value_col='Litho_Code',
                  well_loc_file=well_loc_file,
                  dp=None, fill_value=-1, init_value=1e30,
                  output_file=output_file,
                  tight_frame=True)


##
# Demonstration of using FSDI to interpolate lithology in cube.
from cube import *

multi_file = False
if multi_file:
    seismic_file = ['/nfs/opendtect-data/Niuzhuang/Export/seismic_east.sgy',
                    '/nfs/opendtect-data/Niuzhuang/Export/vpvs_east.sgy',
                    '/nfs/opendtect-data/Niuzhuang/Export/sp_east.sgy']
else:
    seismic_file = '/nfs/opendtect-data/Niuzhuang/Export/seismic_east.sgy'
log_dir = '/nfs/opendtect-data/Niuzhuang/Well logs/LithoCodeForPetrel-time'  # Well log directory.
well_location_file = '/nfs/opendtect-data/Niuzhuang/Well logs/well_locations.prn'  # Well location file.
weight = [5, 5, 5, 1]
log_name = 'Litho_Code'
depth_name = 'TWT'
coord_name = ['X', 'Y']
if multi_file:
    seis_name = ['SeisAmp', 'VpVs', 'SP']
else:
    seis_name = 'SeisAmp'
output_file = '/nfs/opendtect-data/Niuzhuang/Litho_Code_8.txt'
well_name_loc = 'well_name'
coord_name_loc = ['well_X', 'well_Y']
result = FSDI_cube(seismic_file=seismic_file, log_dir=log_dir, output_file=output_file, weight=weight,
                   resample_method='most_frequent', log_name=log_name, depth_name=depth_name, coord_name=coord_name,
                   seis_name=seis_name,  well_location_file=well_location_file,
                   well_name_loc=well_name_loc, coord_name_loc=coord_name_loc)

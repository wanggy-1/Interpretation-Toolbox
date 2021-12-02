# This is my draft.

##
# Obtain data from MySQL database, compute rock-physics parameters and add seismic attributes.
import os
import sys
from well_log import *
from sqlalchemy import create_engine

# Display settings.
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)

# Read well name list in target area.
df_list = pd.read_csv('/nfs/opendtect-data/Niuzhuang/Well logs/wells_in_NiuzhuangEast.txt', names=['WellName', 'Type'],
                      delimiter='\t')
ind = [x for x in range(len(df_list)) if df_list.loc[x, 'Type'] != 'deviated']  # Ignore deviated wells.
well_list = df_list.loc[ind, 'WellName'].values

# Seismic attributes file list and attribute names.
attri_list = ['/nfs/opendtect-data/Niuzhuang/Export/seismic_east.sgy',
              '/nfs/opendtect-data/Niuzhuang/Export/AbsorptionQualityFactor_exp.sgy',
              '/nfs/opendtect-data/Niuzhuang/Export/Energy_exp.sgy',
              '/nfs/opendtect-data/Niuzhuang/Export/Energy_sqrt_exp.sgy',
              '/nfs/opendtect-data/Niuzhuang/Export/Event-Extremum_exp.sgy',
              '/nfs/opendtect-data/Niuzhuang/Export/Event-Maximum_exp.sgy',
              '/nfs/opendtect-data/Niuzhuang/Export/Event-MaximumInGate_exp.sgy',
              '/nfs/opendtect-data/Niuzhuang/Export/Event-Minimum_exp.sgy',
              '/nfs/opendtect-data/Niuzhuang/Export/Event-MinimumInGate_exp.sgy',
              '/nfs/opendtect-data/Niuzhuang/Export/Event-NegativeToPositiveZC_exp.sgy',
              '/nfs/opendtect-data/Niuzhuang/Export/Event-PositiveToNegativeZC_exp.sgy',
              '/nfs/opendtect-data/Niuzhuang/Export/Event-ZeroCrossing_exp.sgy',
              '/nfs/opendtect-data/Niuzhuang/Export/FrequencySlopeFall_exp.sgy',
              '/nfs/opendtect-data/Niuzhuang/Export/InstantaneousAmplitude_exp.sgy',
              '/nfs/opendtect-data/Niuzhuang/Export/InstantaneousFrequency_exp.sgy',
              '/nfs/opendtect-data/Niuzhuang/Export/InstantaneousPhase_exp.sgy',
              '/nfs/opendtect-data/Niuzhuang/Export/SpetralDecomposition10Hz_exp.sgy',
              '/nfs/opendtect-data/Niuzhuang/Export/SpetralDecomposition20Hz_exp.sgy',
              '/nfs/opendtect-data/Niuzhuang/Export/SpetralDecomposition30Hz_exp.sgy',
              '/nfs/opendtect-data/Niuzhuang/Export/SpetralDecomposition40Hz_exp.sgy',
              '/nfs/opendtect-data/Niuzhuang/Export/SpetralDecomposition50Hz_exp.sgy',
              '/nfs/opendtect-data/Niuzhuang/Export/SpetralDecomposition60Hz_exp.sgy',
              '/nfs/opendtect-data/Niuzhuang/Export/SpetralDecomposition70Hz_exp.sgy',
              '/nfs/opendtect-data/Niuzhuang/Export/SpetralDecomposition80Hz_exp.sgy',
              '/nfs/opendtect-data/Niuzhuang/Export/Sweetness_exp.sgy']
attri_name = ['Amp', 'Absorb Q', 'Energy', 'Energy-sqrt', 'Event-extreme', 'Event-max', 'Event-max in gate',
              'Event-min', 'Event-min in gate', 'Event-N2PZC', 'Event-P2NZC', 'Event-ZC', 'Freq Slope Fall',
              'Ins-amp', 'Ins-fre', 'Ins-phase', 'Spectrum-10Hz', 'Spectrum-20Hz', 'Spectrum-30Hz', 'Spectrum-40Hz',
              'Spectrum-50Hz', 'Spectrum-60Hz', 'Spectrum-70Hz', 'Spectrum-80Hz', 'Sweetness']

# Depth-time relation folder.
dt_folder = '/nfs/opendtect-data/Niuzhuang/Well logs TD (new)'

# Output folder.
out_folder = '/nfs/opendtect-data/Niuzhuang/Well logs/TimeLog'

# Connect to SQL database.
con_engine = create_engine('mysql+pymysql://root: @172.19.144.46:3306/mysql')

# Process well by well.
for well in well_list:
    # Print progress.
    print('Processing well %s' % well)

    # Fetch data from database.
    sql_ = f"select * from {well};"
    df_data = pd.read_sql_query(sql_, con_engine)  # Read well log from database.
    df_data.fillna(value=np.nan, inplace=True)  # Fill None with NaN.
    df_data.replace(-999.25, value=np.nan, inplace=True)  # Replace abnormal value with NaN.

    # Compute p-wave velocity from AC log.
    df_data['Vp'] = 1 / df_data['AC'] * 1e3

    # Get depth-time relation file list.
    dt_list = os.listdir(dt_folder)

    # Find depth-time relation.
    for dt_file in dt_list:
        if dt_file[:-4] == well:  # Find the well in depth-time relation list.
            df_dt = pd.read_csv(os.path.join(dt_folder, dt_file), delimiter='\t')
        else:
            continue

        # Remove some columns we don't need.
        log_col = list(df_data.columns)
        remove_col = ['DEPTH', 'well_name', 'well_areaid']  # Remove these columns.
        for col in remove_col:
            log_col.remove(col)

        # Transform well log to time-domain.
        sys.stdout.write('\tDepth-time conversion...')
        df_data = time_log(df_dt=df_dt, df_log=df_data, log_depth_col='DEPTH', dt_depth_col='Depth',
                           time_col='TWT', log_col=log_col, fill_nan=None, delete_nan=False)
        sys.stdout.write('Done.\n')

        # Re-sample to 2ms.
        sys.stdout.write('\tLog resampling...')
        df_data = resample_log(df_log=df_data, delta=2, depth_col='TWT', log_col=log_col, method='nearest',
                               delete_nan=False)
        sys.stdout.write('Done.\n')

        # Get vp/vs ratio from up-hole trace.
        df_data = cube2well(df=df_data, x_col='well_X', y_col='well_Y', z_col='TWT',
                            cube_file='/nfs/opendtect-data/Niuzhuang/Export/vpvs_east.sgy', cube_name='VpVs')

        # Compute rock-physics parameters.
        sys.stdout.write('\tComputing rock-physics parameters...')
        ind = [i for i in range(len(df_data)) if df_data.loc[i, 'VpVs'] == 0]
        df_data.loc[ind, 'VpVs'] = np.nan  # Avoid division by zero.
        df_data['Vs'] = df_data['Vp'] / df_data['VpVs']
        df_data = rock_physics(df=df_data, vp_col='Vp', vs_col='Vs', den_col='DEN')
        sys.stdout.write('Done.\n')

        # Get seismic attributes from up-hole trace.
        for i in range(len(attri_list)):
            sys.stdout.write('\r\tExtracting attributes: %d/%d' % (i+1, len(attri_list)))
            df_data = cube2well(df=df_data, x_col='well_X', y_col='well_Y', z_col='TWT',
                                cube_file=attri_list[i], cube_name=attri_name[i])
        sys.stdout.write(' Done.\n')

        # New column order.
        sys.stdout.write('\tRe-arranging column order...')
        new_col = ['TWT', 'well_X', 'well_Y', 'AC', 'DEN', 'GR', 'SP', 'RT', 'POR', 'PERM', 'SW',
                   'Vp', 'Vs', 'VpVs', 'Ip', 'Is', 'Shear Modulus', 'Bulk Modulus', "Poisson's Ratio",
                   "Young's Modulus"] + attri_name + ['MicroFacies', 'Lith', 'Res']
        df_data = df_data[new_col]
        sys.stdout.write('Done.\n')

        # Write data to file.
        sys.stdout.write('\tSaving to file %s...' % os.path.join(out_folder, well + '.csv'))
        df_data.to_csv(os.path.join(out_folder, well + '.csv'), index=False)
        sys.stdout.write('Done.\n')


##
# Load and visualize marmousi model.
import segyio
import matplotlib.pyplot as plt

# Load marmousi vp, vs and density model.
file_vp = '/nfs/elastic-marmousi-model/model/MODEL_P-WAVE_VELOCITY_1.25m.segy'
file_vs = '/nfs/elastic-marmousi-model/model/MODEL_S-WAVE_VELOCITY_1.25m.segy'
file_den = '/nfs/elastic-marmousi-model/model/MODEL_DENSITY_1.25m.segy'
with segyio.open(file_vp) as f:
    vp = f.xline[0].T
f.close()
with segyio.open(file_vs) as f:
    vs = f.xline[0].T
f.close()
with segyio.open(file_den) as f:
    den = f.xline[0].T
f.close()

plt.figure(1)
plt.title('Vp')
plt.imshow(vp)
plt.colorbar()
plt.figure(2)
plt.title('Vs')
plt.imshow(vs)
plt.colorbar()
plt.figure(3)
plt.title('Density')
plt.imshow(den)
plt.colorbar()
plt.show()


##
# Read LAS log file.
import lasio
import os
from well_log import *

folder = '/nfs/shengli2020/Niuzhuang/Well logs/WellLog/Resistivity/'
subfolder_list = os.listdir(folder)
for subfolder in subfolder_list:
    filelist = os.listdir(os.path.join(folder, subfolder))
    for file in filelist:
        if '.las' in file:
            print('Processing %s' % os.path.join(folder + subfolder, file))
            las = lasio.read(os.path.join(folder + subfolder, file))
            delta = las.well.STEP.value  # Get well log sampling interval.
            df = las.df().reset_index()
            if delta != 0.125:
                depth_col = df.columns[0]
                log_col = df.columns[1:]
                df = resample_log(df_log=df, delta=0.125, depth_col=depth_col, log_col=log_col, method='nearest',
                                  delete_nan=False, fill_nan=None)
                if '(0.1)' in file:
                    file = file.replace('0.1', '0.125')
            df.fillna(-999.25, inplace=True)
            df = df.astype('float32')
            print(df)
            df.to_csv(os.path.join(folder + subfolder, file[:-4] + '.txt'), index=False, sep='\t', float_format='%.3f')


##
# Remove files with certain string in folder.
import os

folder = '/nfs/shengli2020/Niuzhuang/Well logs/WellLog/Resistivity/'
subfolder_list = os.listdir(folder)
for subfolder in subfolder_list:
    filelist = os.listdir(os.path.join(folder, subfolder))
    for file in filelist:
        if '.txt' in file:  # Only remove txt file.
            print('Remove file %s' % os.path.join(folder + subfolder, file))
            os.remove(os.path.join(folder + subfolder, file))


##
# Assemble data set and select the most informative features.
import os
from well_log import *
from machine_learning import *

pd.set_option('display.max_columns', 20)

# Assemble data.
folder = '/nfs/opendtect-data/Niuzhuang/Well logs/TimeLog'
df = pd.DataFrame()
for file in os.listdir(folder):
    df_temp = pd.read_csv(os.path.join(folder, file))
    df = df.append(df_temp, ignore_index=True)
print('Original data frame:\n', df)

# Select features and target.
target = 'Res'
rm_col = ['TWT', 'well_X', 'well_Y', 'MicroFacies', 'Lith', 'Res']
new_col = list(df.columns)
for col in rm_col:
    if col != target:
        new_col.remove(col)
df_xy = df[new_col].copy()
print('Feature and target data frame:\n', df_xy)
check_info(df_xy, log_col=list(df_xy.columns))

# Deal with abnormal values in each feature.
con = {'AC': [130, None], 'PERM': [0, 100], 'Vp': [None, 8], 'Vs': [None, 7], 'Ip': [None, 22], 'Is': [None, 20],
       'Bulk Modulus': [0, None]}
df_xy = outlier_filter(df_xy, condition=con, delete_inf=True, delete_none=True, remove_row=False)
print('Feature and target data frame after outlier filter:\n', df_xy)
check_info(df_xy, log_col=list(df_xy.columns))

# Filter out features which are highly correlated to another feature.
threshold = 0.9
df_xy = high_cor_filter(df=df_xy, threshold=threshold, annot=True, axis_tick_size=12, cbar_tick_size=14,
                        cbar_label_size=16, title_size=20)
print('Feature and target data frame after HFC:\n', df_xy)
check_info(df_xy, log_col=list(df_xy.columns))

# Assemble dataset.
df_xy.dropna(axis='index', how='any', subset=[target], inplace=True)  # Remove rows with NaN labels.
df_xy.reset_index(drop=True, inplace=True)
df_xy.dropna(axis='columns', how='any', inplace=True)  # Remove columns with NaN.
feature = list(df_xy.columns)
feature.remove(target)
print('Features remain:\n', feature)
df_xy[target] = df_xy[target].astype('int32')
print('Feature and target data frame before RFE:\n', df_xy)
check_info(df_xy, log_col=list(df_xy.columns))

# Select features.
df_xy, _, _ = feature_selection(df=df_xy, feature_col=feature, target_col=target,
                                estimator_type='classifier',
                                auto=False, random_state=0, n_features_to_select=5)
print('Final feature and target data frame:\n', df_xy)
check_info(df_xy, log_col=list(df_xy.columns))


##
# Use agglomerative clustering to cluster lithology and micro facies.
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from horizon import *
from matplotlib.colors import LinearSegmentedColormap

# Display more columns.
pd.set_option('display.max_columns', 20)

# Folder and horizon file name.
folder = '/nfs/opendtect-data/Niuzhuang/Export'  # Folder name to get horizon data.
log_folder = '/nfs/opendtect-data/Niuzhuang/Well logs/LithoCodeForPetrel-time'  # Folder name to get log data.
output_folder = '/nfs/opendtect-data/Niuzhuang'  # Folder name to save results.
picture_folder = '/nfs/opendtect-data/Niuzhuang/Pictures'  # Folder name to save images.
well_xy_file = '/nfs/opendtect-data/Niuzhuang/Well logs/well_locations.prn'  # Well location file.
hor_list = ['z1', 'z2', 'z3', 'z4', 'z5', 'z6']  # Horizon names.
target = 'Litho_Code'  # Clustering target name.
n_clusters = 2  # The number of clusters.
cmap_color = ['tab:blue', 'tab:orange']  # Colors of the clustering result.
labels = ['0', '1']  # Label names of the clustering result.
class_code = [0, 1]  # Label codes of the clustering result.
marker_class_code = {'Litho_Code': [0, 1],  # Class codes of markers
                     'MicroFacies': [1, 2, 3, 4, 5, 6, 7]}
marker_color = {'Litho_Code': ['grey', 'gold'],  # Colors of markers.
                'MicroFacies': ['r', 'g', 'b', 'c', 'm', 'y', 'k']}
marker_labels = {'Litho_Code': ['Mudstone', 'Sandstone'],  # Labels of markers.
                 'MicroFacies': ['Diversion channel', 'Diversion bay', 'Submarine diversion channel',
                                 'Submarine diversion bay', 'Estuary dam', 'Semi-deep lake mud', 'Turbidite']}
suffix = '-features-clustering.dat'

# The selected features of different targets.
selected_features = {'Litho_Code': ['sp', 'vpvs', 'Spectrum-70Hz', 'Absorb Q', 'Spectrum-60Hz'],
                     'MicroFacies': ['sp', 'vpvs', 'Absorb Q', 'Spectrum-60Hz', 'Event-min']}

# Get well location.
df_well_xy = pd.read_csv(well_xy_file, delimiter='\s+')

for hor in hor_list:
    # Get horizon data.
    df_hor = pd.read_csv(os.path.join(folder, hor+suffix), delimiter='\t')

    # Get features for the selected target.
    df_feature = df_hor[selected_features[target]].copy()
    x = df_feature.values

    # Scale all features to range [0, 1].
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    # Agglomerative clustering.
    sys.stdout.write('Clustering %s on horizon %s...' % (target, hor))
    t1 = time.perf_counter()
    cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')  # Change the number of clusters.
    y = cluster.fit_predict(x)
    t2 = time.perf_counter()
    sys.stdout.write(' Done.\n')
    sys.stdout.write('Clustering time: %.2fs\n' % (t2 - t1))

    # Make a data frame of the clustering result.
    df_result = df_hor[['X', 'Y', 'Z'] + selected_features[target]].copy()
    df_result[target] = y
    df_result[target] = df_result[target].astype('int')
    sys.stdout.write('Saving result to file %s...' %
                     (os.path.join(output_folder, f'{hor}-{target}' + '-Clustering(unsupervised).dat')))
    df_result.to_csv(os.path.join(output_folder, f'{hor}-{target}' + '-Clustering(unsupervised).dat'), sep='\t',
                     index=False)
    sys.stdout.write(' Done.\n')

    # Create markers.
    marker = horizon_log(df_horizon=df_result[['X', 'Y', 'Z']], df_well_coord=df_well_xy, log_file_path=log_folder,
                         log_x_col='well_X', log_y_col='well_Y', log_z_col='TWT', log_value_col=target,
                         log_abnormal_value=-999, sep='\t',
                         horizon_x_col='X', horizon_y_col='Y', horizon_z_col='Z',
                         print_progress=False)
    if target == 'Litho_Code':  # Binarize lithology codes.
        marker.loc[marker[target] <= 1, target] = 0
        marker.loc[marker[target] > 1, target] = 1

    # Visualize the clustering result.
    cm = LinearSegmentedColormap.from_list('custom', cmap_color, len(cmap_color))
    visualize_horizon(df=df_result, x_name='X', y_name='Y', value_name=target, cmap=cm, nominal=True,
                      class_code=class_code, class_label=labels, vmin=min(class_code), vmax=max(class_code),
                      fig_name=f'{hor}-{target}' + '-Clustering(unsupervised)', show=False)
    plot_markers(df=marker, x_col='X', y_col='Y', class_col=target, wellname_col='WellName',
                 class_code=marker_class_code[target], class_label=marker_labels[target], colors=marker_color[target])
    plt.savefig(os.path.join(picture_folder, f'{hor}-{target}' + '-Clustering(unsupervised).png'), dpi=300)

    # Visualize seismic data.
    visualize_horizon(df=df_hor, x_name='X', y_name='Y', value_name='seismic', nominal=False, fig_name=hor + '-Seismic',
                      show=False)
    plot_markers(df=marker, x_col='X', y_col='Y', class_col=target, wellname_col='WellName',
                 class_code=marker_class_code[target], class_label=marker_labels[target], colors=marker_color[target])
    # plt.savefig(os.path.join(picture_folder, hor + '-Seismic with ' + target + ' 3 classes.png'), dpi=300)


##
# Visualize horizons and plot markers on horizons.
from horizon import *
from matplotlib.colors import LinearSegmentedColormap

pd.set_option('display.max_column', 20)

hor_list = ['z1', 'z2', 'z3', 'z4', 'z5', 'z6']
hor_folder = '/nfs/opendtect-data/Niuzhuang'
hor_suffix = 'Clustering(unsupervised).dat'
log_folder = '/nfs/opendtect-data/Niuzhuang/Well logs/LithoCodeForPetrel-time'
well_xy_file = '/nfs/opendtect-data/Niuzhuang/Well logs/well_locations.prn'
output_folder = '/nfs/opendtect-data/Niuzhuang/Pictures'
target = 'Litho_Code'
output_suffix = 'Clustering(unsupervised, no markers)'
binarize = True
plot_marker = False
nominal = True
save_fig = True

if binarize:
    marker_color = ['grey', 'gold']  # Two lithology.
    labels = ['0', '1']  # Two lithology.
    class_code = [0, 1]  # Two lithology.
else:
    labels = ['mudstone', 'lime-mudstone', 'siltstone', 'sandstone', 'gravel-sandstone']  # Five lithology.
    class_code = [0, 1, 2, 3, 4]  # Five lithology.
    marker_color = ['grey', 'limegreen', 'cyan', 'gold', 'darkviolet']  # Five lithology.

for hor in hor_list:
    df_horizon = pd.read_csv(os.path.join(hor_folder, f'{hor}-{target}-{hor_suffix}'), delimiter='\t')
    df_well_xy = pd.read_csv(well_xy_file, delimiter='\s+')

    # if binarize:
    #     df_horizon.loc[df_horizon[target] <= 1, target] = 0
    #     df_horizon.loc[df_horizon[target] > 1, target] = 1

    if plot_marker:
        marker = horizon_log(df_horizon=df_horizon[['X', 'Y', 'Z']], log_file_path=log_folder, df_well_coord=df_well_xy,
                             sep='\t', log_x_col='well_X', log_y_col='well_Y', log_z_col='TWT',
                             log_value_col=target, log_abnormal_value=-999,
                             horizon_x_col='X', horizon_y_col='Y', horizon_z_col='Z',
                             print_progress=False)
        if binarize:
            marker.loc[marker[target] <= 1, target] = 0
            marker.loc[marker[target] > 1, target] = 1
    if nominal:
        cm = LinearSegmentedColormap.from_list('custom', marker_color, len(marker_color))
        visualize_horizon(df=df_horizon, x_name='X', y_name='Y', value_name=target, cmap=cm, nominal=True,
                          class_code=class_code, class_label=labels, vmin=min(class_code), vmax=max(class_code),
                          fig_name=f'{hor}-Clustering')
        if plot_marker:
            plot_markers(df=marker, x_col='X', y_col='Y', class_col=target, wellname_col='WellName',
                         class_label=labels, colors=marker_color, class_code=class_code)
    else:
        visualize_horizon(df=df_horizon, x_name='X', y_name='Y', value_name=target, cmap='rainbow',
                          fig_name=f'{hor}-{target}')
    if save_fig:
        plt.savefig(f'/nfs/opendtect-data/Niuzhuang/Pictures/{hor}-{target}-{output_suffix}.png', dpi=300)


##
# Add lithology codes to horizon features.
import pandas as pd
import numpy as np
import os
import sys

pd.set_option('display.max_columns', 20)

lith_folder = '/nfs/opendtect-data/Niuzhuang'
feature_folder = '/nfs/opendtect-data/Niuzhuang/Export'
hor_list = ['z2', 'z3', 'z4', 'z5', 'z6']
lith_suffix = '-Lithology.dat'
feature_suffix = '-features-clustering.dat'

for hor in hor_list:
    sys.stdout.write('Processing on horizon %s...' % hor)
    df_lith = pd.read_csv(os.path.join(lith_folder, hor + lith_suffix), delimiter='\t')
    df_col = pd.read_csv(os.path.join(feature_folder, hor + feature_suffix), nrows=0, delimiter='\t')
    col = list(df_col.columns)
    col[0] = 'X'
    df_feature = pd.read_csv(os.path.join(feature_folder, hor + feature_suffix), skiprows=2, delimiter='\t', names=col)
    df_feature['Lith'] = df_lith['Litho_Code'].values
    df_feature['Lith'] = df_feature['Lith'].astype(np.int64)
    sys.stdout.write(' Done.\n')
    sys.stdout.write('Saving result to file %s...' % (os.path.join(feature_folder, hor + feature_suffix)))
    df_feature.to_csv(os.path.join(feature_folder, hor + feature_suffix), index=False, sep='\t')
    sys.stdout.write(' Done.\n')


##
# Change the column configuration in data frame.
import pandas as pd

pd.set_option('display.max_columns', 20)

hor_list = ['z1', 'z2', 'z3', 'z4', 'z5', 'z6']
for hor in hor_list:
    print('Processing horizon %s' % hor)
    filename = f'/nfs/opendtect-data/Niuzhuang/Export/{hor}-features-clustering.dat'
    df = pd.read_csv(filename, delimiter='\t', nrows=0)
    col = list(df.columns)
    col[0] = 'X'
    df = pd.read_csv(filename, delimiter='\t', skiprows=2, names=col)
    df.to_csv(filename, index=False, sep='\t')
    print('Done.')


##
# Generate pseudo-horizons.
import pandas as pd
import numpy as np

col_name = ['X', 'Y', 'INLINE', 'XLINE', 'Z']
dp = 0.2

p = np.arange(dp, 1, dp)
print(p)

horizon_1 = pd.read_csv('/nfs/opendtect-data/Niuzhuang/Export/z4_east_dense.dat', delimiter='\t', names=col_name)
horizon_2 = pd.read_csv('/nfs/opendtect-data/Niuzhuang/Export/z5_east_dense.dat', delimiter='\t', names=col_name)

print((horizon_1[['X', 'Y']] == horizon_2[['X', 'Y']]).all())

th = horizon_2['Z'] - horizon_1['Z']

for i in range(len(p)):
    horizon_new = horizon_1.copy()
    horizon_new['Z'] = horizon_1['Z'] + p[i] * th
    horizon_new.to_csv(f'/nfs/opendtect-data/Niuzhuang/Export/z4-z5_east_dense-{p[i]}.dat', sep='\t', index=False,
                       header=False)


##
# Visualize cube data.
from cube import *

file = '/nfs/opendtect-data/Niuzhuang/Export/seismic_east.sgy'
hor_file = ['/nfs/opendtect-data/Niuzhuang/z1_dense_full.dat']
plot_cube(cube_file=file, value_name='seismic', scale=[10, 10, 1], show_axes=True, on='ix',
          hor_list=hor_file, hor_header=None, hor_col_names=['INLINE', 'XLINE', 'X', 'Y', 'Z'],
          hor_x='X', hor_y='Y', hor_il='INLINE', hor_xl='XLINE', hor_z='Z')


##
# Get data from cube to horizon.
from horizon import *

cube_file = '/nfs/opendtect-data/Niuzhuang/Export/seismic_east.sgy'
horizon_file = '/nfs/opendtect-data/Niuzhuang/Horizons/z1_dense.dat'

df_horizon = pd.read_csv(horizon_file, delimiter='\t', names=['INLINE', 'XLINE', 'X', 'Y', 'Z'], header=None)
df_horizon = cube2horizon(df_horizon=df_horizon, cube_file=cube_file, hor_x='X', hor_y='Y', hor_il='INLINE',
                          hor_xl='XLINE', hor_z='Z', match_on='ix', value_name='seismic')
visualize_horizon(df=df_horizon, x_name='INLINE', y_name='XLINE', value_name='seismic', nominal=False,
                  deltax=1, deltay=1)


##
# Interpolate a horizon.
from horizon import *

hor_file = '/nfs/opendtect-data/Niuzhuang/Horizons/z1_dense.dat'
df_hor = pd.read_csv(hor_file, delimiter='\t', names=['INLINE', 'XLINE', 'X', 'Y', 'Z'])
print(df_hor)
print('NaN: %d' % df_hor.isna().sum().sum())
df_new = horizon_interp(df=df_hor, x_col='INLINE', y_col='XLINE', t_col='Z', x_step=1, y_step=1, visualize=False)
print(df_new)
print('NaN: %d' % df_new.isna().sum().sum())
df_hor['Z'] = df_new['Z'].values
print(df_hor)
print('NaN: %d' % df_hor.isna().sum().sum())
df_hor.to_csv('/nfs/opendtect-data/Niuzhuang/z1_dense_full.dat', sep='\t', index=False, header=False)

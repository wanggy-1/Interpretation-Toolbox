import time
import warnings
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, RFECV
from sklearn.cluster import AgglomerativeClustering


def high_cor_filter(df=None, threshold=0.9, cor_method='pearson', cor_vis=False, cmap='Reds', annot=True, fmt='.2f',
                    vmin=None, vmax=None, annot_size=None, axis_tick_size=None,
                    cbar_tick_size=None, cbar_label_size=None, title_size=None):
    """
    Remove features which are highly correlated with other features.
    https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python
    :param df: (Pandas.Dataframe) - Feature data frame.
    :param threshold: (Float) - Default is 0.9. Filter threshold, feature with correlation higher than this threshold
                      will be removed.
    :param cor_method: (String) - Default is 'pearson', the standard correlation coefficient.
                       'kendall': Kendall Tau correlation coefficient.
                       'spearman': Spearman rank correlation.
    :param cor_vis: (Bool) - Default is False. Whether to visualize the correlation matrix with heatmap.
    :param cmap: (String) - Color map of the heatmap.
    :param annot: (Bool) - Default is True. Whether to write correlation value in each cell on the heatmap.
    :param fmt: (String) - Default is '.2f'. String formatting code to use when adding annotations.
    :param vmin: (Float) - Default is None, which is inferred from the data. Minimum value of the color map.
    :param vmax: (Float) - Default is None, which is inferred from the data. Maximum value of the color map.
    :param annot_size: (Integer) - Default is None, which is self-adapted to the figure. The annotation font size of
                       the heatmap.
    :param axis_tick_size: (Integer) - Default is None, which is self-adapted to the figure. The x and y axis tick label
                           size of the heatmap.
    :param cbar_tick_size: (Integer) - Default is None, which is self-adapted to the figure. The color bar tick size of
                           the heatmap.
    :param cbar_label_size: (Integer) - Default is None, which is self-adapted to the figure. The color bar label size
                            of the heatmap.
    :param title_size: (Integer) - Default is None, which is self-adapted to the figure. The title size of the heatmap.
    :return: df: (Pandas.Dataframe) - Filtered feature data frame.
    """
    # Scale all features to 0~1.
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_new = scaler.fit_transform(df.values)
    df_new = pd.DataFrame(x_new, columns=df.columns)
    # Compute absolute correlation matrix between features.
    cor = df_new.corr(method=cor_method).abs()
    # If cor_vis is True, visualize correlation matrix as a heatmap.
    plt.figure(figsize=[16, 16])
    plt.title('Correlation Matrix - Before High Correlation Filter', fontsize=title_size)
    ax = sns.heatmap(cor, annot=annot, cmap=cmap, xticklabels=1, yticklabels=1, fmt=fmt, vmin=vmin, vmax=vmax,
                     annot_kws={'size': annot_size})
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=axis_tick_size)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=axis_tick_size)
    cbar = ax.collections[0].colorbar
    cbar.set_label('Correlation', size=cbar_label_size)
    cbar.ax.tick_params(labelsize=cbar_tick_size)
    # Filter out features which are highly correlated with another feature.
    cor_u = cor.where(np.triu(np.ones(cor.shape), k=1).astype(bool))  # The correlation matrix is symmetrical.
    drop_col = [col for col in cor_u.columns if any(cor_u[col] > threshold)]  # Columns with correlation > threshold.
    print('%d features are removed by HCF' % len(drop_col))
    print(drop_col)
    df_new.drop(columns=drop_col, inplace=True)  # In this data frame all features are scaled to 0~1.
    df.drop(columns=drop_col, inplace=True)  # This is the original data frame with unscaled features.
    # If cor_vis is True, visualize correlation matrix after dropping out features with high correlation.
    cor_new = df_new.corr(method=cor_method).abs()
    plt.figure(figsize=[16, 16])
    plt.title('Correlation Matrix - After High Correlation Filter', fontsize=title_size)
    ax = sns.heatmap(cor_new, annot=annot, cmap=cmap, xticklabels=1, yticklabels=1, fmt=fmt, vmin=vmin, vmax=vmax,
                     annot_kws={'size': annot_size})
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=axis_tick_size)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=axis_tick_size)
    cbar = ax.collections[0].colorbar
    cbar.set_label('Correlation', size=cbar_label_size)
    cbar.ax.tick_params(labelsize=cbar_tick_size)
    # Choose whether to display correlation matrix heatmap.
    if cor_vis:
        plt.show()
    return df


def feature_selection(df=None, feature_col=None, target_col=None, random_state=None, estimator_type='classifier',
                      auto=True, n_features_to_select=None, show=True):
    """
    Select most informative features by recursive feature elimination (RFE) or RFE in a cross-validation loop (RFECV).
    https://scikit-learn.org/stable/modules/feature_selection.html#rfe
    The RFE selects features by recursively considering smaller and smaller sets of features until it reaches the pre-
    defined number. The RFECV performs RFE in a cross-validation loop to find the optimal number of features.
    Here the Random Forests classifier and the Random Forests regressor are used as base estimator for classification
    and regression respectively.
    :param df: (Pandas.Dataframe) - Data frame that contains features and the target variable.
    :param feature_col: (List of strings or string) - Feature column names in df.
    :param target_col: (String) - Target variable column name in df.
    :param random_state: (Integer or None) - Default is 0. The random number seed. If None, will produce different
                         results in different calls. If integer, will produce same results in different calls.
    :param estimator_type: (String) - Default is 'classifier'. The estimator type, 'classifier' to use the Random
                           Forests classifier and 'regressor' to use the Random Forests regressor.
    :param auto: (Bool) - Default is True. Whether to automatically select the optimal number of features. If true, will
                 use the RFECV, otherwise will use the RFE and require to input the disired number of selected features.
    :param n_features_to_select: (Integer, float or None) - Default is None. The number of features to select.
                                 Only used when auto is False. If None, half of the features are selected.
                                 If integer, the parameter is the absolute number of features to select.
                                 If float between 0 and 1, it is the fraction of features to select.
    :param show: (Bool) - Default is True.
                 Whether to show the feature rank, feature importance and cross-validation (CV) accuracy curve.
                 To show CV accuracy curve, RFECV must be used.
    :return: df_out: (Pandas.Dataframe) - Data frame with selected features and the target variable.
             df_rank: (Pandas.Dataframe) - The ranking of all features.
             df_importance: (Pandas.Dataframe) - The importance of selected features.
    """
    warnings.simplefilter('ignore')  # The warning is annoying...
    # Set estimator.
    if estimator_type == 'classifier':
        estimator = RandomForestClassifier(random_state=random_state)
    elif estimator_type == 'regressor':
        estimator = RandomForestRegressor(random_state=random_state)
    else:
        raise ValueError("estimator_type can either be 'classifier' or 'regressor'.")
    # Get features and target.
    x = df[feature_col].copy()
    y = df[target_col].copy()
    # If auto is True, select the optimal number of features by RFE in cross-validation loop.
    if auto:
        selector = RFECV(estimator, cv=5, step=1)
    # If auto is False, select the user-defined number of features by recursive feature elimination (RFE).
    else:
        selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
    # Fit selector with features and target.
    selector.fit(x, y)
    # Print selected features.
    print('%d features are selected by RFE:' % selector.n_features_)
    selected_feature = list(selector.get_feature_names_out())
    print(selected_feature)
    # Print the feature rank.
    df_rank = pd.DataFrame({'Rank': selector.ranking_}, index=selector.feature_names_in_)
    df_rank.sort_values(by=['Rank'], inplace=True)
    if show:
        print('The rank of all features:\n', df_rank)
    # Print feature importance.
    feature_importance = selector.estimator_.feature_importances_
    df_importance = pd.DataFrame({'Importance': feature_importance}, index=selected_feature)
    df_importance.sort_values(by=['Importance'], ascending=False, inplace=True)
    if show:
        print('Feature importance:\n', df_importance)
    # Dataframe with selected features.
    df_out = pd.DataFrame(selector.transform(x), columns=selected_feature)
    df_out[target_col] = df[target_col].values
    # If automatically select the optimal number of features, draw a curve of cross-validation accuracy.
    if auto:
        plt.figure(figsize=[12, 8])
        plt.style.use('bmh')
        plt.title('REFCV Result', fontsize=20)
        plt.xlabel('Number of feature selected', fontsize=18)
        plt.ylabel('Cross validation score (accuracy)', fontsize=18)
        plt.tick_params(labelsize=14)
        plt.xticks(range(selector.n_features_in_ + 1))
        plt.plot(range(1, selector.n_features_in_ + 1), selector.cv_results_['mean_test_score'], 'ro--', lw=2,
                 markeredgecolor='k', ms=8)
        plt.errorbar(range(1, selector.n_features_in_ + 1), selector.cv_results_['mean_test_score'],
                     yerr=selector.cv_results_['std_test_score'], fmt='none', ecolor='k', capsize=5, elinewidth=1)
        if show:
            plt.show()
    return df_out, df_rank, df_importance


def agglomerative_clustering(df_feature=None, n_cluster=2, affinity='euclidean', linkage='ward', **kwargs):
    """
    Use agglomerative clustering to cluster features.
    :param df_feature: (Pandas.Dataframe) - Feature dataframe.
    :param n_cluster: (Integer) - The number of clusters to find. It must be None if distance_threshold is not None.
    :param affinity: (String) - Default is 'euclidean'. Metric used to compute the linkage. Can be 'euclidean', 'l1',
                     'l2', 'manhattan', 'cosine', or 'precomputed'. If linkage is 'ward', only 'euclidean' is accepted.
                     If 'precomputed', a distance matrix (instead of a similarity matrix) is needed as input for the
                     fit method.
    :param linkage: (String) - Default is 'ward'. Which linkage criterion to use.The linkage criterion determines which
                    distance to use between sets of observation. The algorithm will merge the pairs of cluster that
                    minimize this criterion.
                    Options are 'ward', 'complete', 'average', 'single'.
                    'ward' minimizes the variance of the clusters being merged.
                    'average' uses the average of the distances of each observation of the two sets.
                    'complete' or 'maximum' linkage uses the maximum distances between all observations of the two sets.
                    'single' uses the minimum of the distances between all observations of the two sets.
    :param kwargs: (Dictionary) - Keyword parameters of function AgglomerativeClustering from scikit-learn.
                   https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
    :return: df_out: (Pandas.DataFrame) - Features and cluster labels. The cluster label column name is 'class'.
             clustering: (Object) - The fitted instance of AgglomerativeClustering.
    """
    t1 = time.perf_counter()  # Timer.
    # Get features as array.
    x = df_feature.values
    # Scale all features to 0 and 1.
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    # Start clustering.
    sys.stdout.write('Clustering...')
    clustering = AgglomerativeClustering(n_clusters=n_cluster, affinity=affinity, linkage=linkage, **kwargs)
    y = clustering.fit_predict(x)
    sys.stdout.write('Done\n')
    # Output.
    df_out = df_feature.copy()
    df_out['class'] = y
    t2 = time.perf_counter()
    print('Process time: %.2fs' % (t2 - t1))
    return df_out, clustering

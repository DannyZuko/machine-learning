import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics, linear_model, svm, neighbors
# from sklearn import metrics
# from sklearn import linear_model
from datetime import timedelta

data = pd.read_csv('WIKI-AAPL.csv', index_col=0, parse_dates=True,
                   infer_datetime_format=True)
volumes = data['adj_volume']

n_obs = volumes.count()
mean = volumes.mean()
median = volumes.median()
std = volumes.std()
skew = volumes.skew()
kurt = volumes.kurt()
min = volumes.min()
argmin = volumes.argmin()
max = volumes.max()
argmax = volumes.argmax()

q3, q1 = np.percentile(volumes, [75, 25])
iqr = q3 - q1
outlier_lb = q1 - iqr * 1.5
outlier_ub = q3 + iqr * 1.5
major_outlier_lb = q1 - iqr * 3
major_outlier_ub = q3 + iqr * 3
is_outlier = (volumes < outlier_lb) | (volumes > outlier_ub)
is_major_outlier = (volumes < major_outlier_lb) | (volumes > major_outlier_ub)
n_outliers = is_outlier.sum()
n_major_outliers = is_major_outlier.sum()
n_minor_outliers = n_outliers - n_major_outliers

# box and whisker plots
# showing all outliers
plt.boxplot(volumes, vert=False, labels=['Volumes'], showmeans=True)
plt.title('Box-and-whisker plots including all outliers')
# ignoring major outliers only
plt.boxplot(volumes, vert=False, labels=['Volumes'], showmeans=True,
            showfliers=False, whis=3)
plt.title('Box-and-whisker plots excluding major outliers only')
# ignoring all outliers
plt.boxplot(volumes, vert=False, labels=['Volumes'], showmeans=True,
            showfliers=False)
plt.title('Box-and-whisker plots excluding all outliers')

# histogram of volume distribution
plt.hist(volumes, bins=500, histtype='stepfilled')
plt.xlabel('Volume')
plt.ylabel('N. of observations')
plt.title('Histogram of volume distribution')

# histogram of log(volume) distribution
plt.hist(np.log(volumes), bins=100, histtype='stepfilled')
plt.xlabel('Volume')
plt.ylabel('N. of observations')
plt.title('Histogram of log(volume) distribution')

data = data.reindex(index=data.index[::-1])
data = data[['adj_close', 'adj_volume']]

daily_returns = data['adj_close'] / data['adj_close'].shift(1) - 1
daily_returns = pd.DataFrame(daily_returns)
daily_returns.columns = ['daily_returns']
data = data.join(daily_returns)

cumulative_returns = data['adj_close'] / data['adj_close'][0] - 1
cumulative_returns = pd.DataFrame(cumulative_returns)
cumulative_returns.columns = ['cumulative_returns']
data = data.join(cumulative_returns)

# plot line chart of cumulative returns
# remember to put dates on x axis
plt.plot(data['cumulative_returns'].values)

# for every dollar invested in Apple stock on 12th Dec 1980 (beginning of the
# period), you would made a profit of 249.29$
final_pnl = data['cumulative_returns'][-1]

# avg annual return = avg daily return * number of trading days in a year
avg_annual_return = data['daily_returns'].mean() * 252

# avg annual std = daily std * sqrt(number of trading days in a year)
# we take the sqrt because
# std = sqrt(variance)
# annual variance = daily variance * 252
# annual std = sqrt(annual variance)
#            = sqrt(daily variance * 252)
#            = sqrt(daily variance) * sqrt(252)
#            = daily std * sqrt(252)
annual_std = data['daily_returns'].std() * np.sqrt(252)

cum_ret = data['cumulative_returns']
drawdowns = np.maximum.accumulate(cum_ret) - cum_ret
max_drawdown_valley = np.argmax(drawdowns)
max_drawdown_peak = np.argmax(cum_ret[:max_drawdown_valley])
max_drawdown = np.max(drawdowns)

plt.plot(cum_ret)
plt.plot([max_drawdown_peak, max_drawdown_valley],
         [cum_ret[max_drawdown_peak], cum_ret[max_drawdown_valley]],
         'o', color='Red', markersize=10)

avg_drawdown = np.mean(drawdowns)
median_drawdown = np.median(drawdowns)
q3_drawdown = np.percentile(drawdowns, 75)

# remember to compute recovery time from drawdowns

# risk-adjusted return measures
# all such measures are annualized here
# information ratio = sharpe ratio without taking risk-free rate into account
# using Narang's definition here, not Balch's - hence ignoring beta
information_ratio = avg_annual_return / annual_std

# sterling ratio = avg return / std of below average returns
below_avg_returns = daily_returns[daily_returns < daily_returns.mean()]
annual_below_avg_returns_std = below_avg_returns.std() * np.sqrt(252)
sterling_ratio = avg_annual_return / annual_below_avg_returns_std
sterling_ratio = sterling_ratio.values[0]

# adjusted sterling ratio = avg return / std of negative returns
negative_returns = daily_returns[daily_returns < 0]
annual_negative_returns_std = negative_returns.std() * np.sqrt(252)
adjusted_sterling_ratio = avg_annual_return / annual_negative_returns_std
adjusted_sterling_ratio = adjusted_sterling_ratio.values[0] 

calmar_ratio = avg_annual_return / max_drawdown

# omega ratio = sum of all positive returns / sum of all negative returns
positive_returns = daily_returns[daily_returns > 0]
omega_ratio = positive_returns.sum() / negative_returns.sum().abs()
omega_ratio = omega_ratio.values[0]

metrics_values = [final_pnl, avg_annual_return, annual_std, max_drawdown,
                  avg_drawdown, median_drawdown, information_ratio,
                  sterling_ratio, adjusted_sterling_ratio, calmar_ratio,
                  omega_ratio]

metrics_labels = ['final_pnl', 'avg_annual_return', 'annual_std', 'max_drawdown',
                  'avg_drawdown', 'median_drawdown', 'information_ratio',
                  'sterling_ratio', 'adjusted_sterling_ratio', 'calmar_ratio',
                  'omega_ratio']

metrics_df = pd.DataFrame(metrics_values, metrics_labels, ['value'])

# scale momentum values in order to have all values within range [-1; 1]
# use MinMax scaler from sklearn on abs(momentum)
# then multiply scaled values by sign(momentum) so that scaled values have same
# sign as non-scaled values
minmax_scaler = preprocessing.MinMaxScaler()

momentum = data['adj_close'] / data['adj_close'].shift(5) - 1
momentum = pd.DataFrame(momentum)
momentum.columns = ['momentum']
data = data.join(momentum)

momentum_dropna = momentum.dropna()
scaled_momentum = minmax_scaler.fit_transform(abs(momentum_dropna)) * \
                  np.sign(momentum_dropna) 

volumes = data['adj_volume']
volumes = pd.DataFrame(volumes)
scaled_volumes = minmax_scaler.fit_transform(volumes)

momentum_by_volume = momentum * volumes.values
momentum_by_volume.columns = ['momentum_by_volume']
data = data.join(momentum_by_volume)

scaled_momentum_by_volume = \
minmax_scaler.fit_transform(momentum_by_volume.dropna()) 


plt.boxplot(data['daily_returns'].dropna().values,
	    vert=False,
	    labels=['daily_returns'],
	    showmeans=True)
plt.title('Box-and-whisker plot of daily returns including all outliers')


plt.boxplot(momentum.dropna().values,
	    vert=False,
	    labels=['momentum'],
	    showmeans=True)
plt.title('Box-and-whisker plot of momentum including all outliers')


plt.boxplot(volumes.values,
	    vert=False,
	    labels=['volumes'],
	    showmeans=True)
plt.title('Box-and-whisker plot of volumes including all outliers')


plt.boxplot(np.log(volumes).values,
	    vert=False,
	    labels=['log_volumes'],
	    showmeans=True)
plt.title('Box-and-whisker plot of log(volumes) including all outliers')


plt.boxplot(momentum_by_volume.dropna().values,
	    vert=False,
	    labels=['momentum_by_volume'],
	    showmeans=True)
plt.title('Box-and-whisker plot of momentum_by_volume including all outliers')

# maybe want to plot momentum * log(volumes) ?


plt.hist(data['daily_returns'].dropna().values, bins=200, histtype='stepfilled')
plt.xlabel('daily_returns')
plt.ylabel('N. of observations')
plt.title('Histogram of daily_returns distribution')


plt.hist(momentum.dropna().values, bins=200, histtype='stepfilled')
plt.xlabel('momentum')
plt.ylabel('N. of observations')
plt.title('Histogram of momentum distribution')


plt.hist(volumes.values, bins=500, histtype='stepfilled')
plt.xlabel('volumes')
plt.ylabel('N. of observations')
plt.title('Histogram of volume distribution')


plt.hist(np.log(volumes).values, bins=500, histtype='stepfilled')
plt.xlabel('volumes')
plt.ylabel('N. of observations')
plt.title('Histogram of volume distribution')


plt.hist(momentum_by_volume.dropna().values, bins=1000, histtype='stepfilled')
plt.xlabel('momentum_by_volume')
plt.ylabel('N. of observations')
plt.title('Histogram of momentum_by_volume distribution')


delta_volume = data['adj_volume'] / data['adj_volume'].shift(1) - 1
delta_volume = pd.DataFrame(delta_volume)
plt.hist(delta_volume.dropna().values, bins = 1000)



pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')

data_subset = data.loc[:, ['adj_volume',
                           'daily_returns',
                           'momentum',
                           'momentum_by_volume']] 

pd.scatter_matrix(data_subset, alpha = 0.3, figsize = (14,8), diagonal = 'kde')

# explain why excluding columns from scatter matrix


def summary(df):
    '''Provide summary of relevant statistics for some given feature'''
    
    df = df.iloc[:, 0]
    n_obs = df.count()
    mean = df.mean()
    median = df.median()
    std = df.std()
    skew = df.skew()
    kurt = df.kurt()
    min = df.min()
    argmin = df.argmin()
    max = df.max()
    argmax = df.argmax()
    q3, q1 = np.percentile(df.dropna(), [75, 25])
    iqr = q3 - q1
    outlier_lb = q1 - iqr * 1.5
    outlier_ub = q3 + iqr * 1.5
    major_outlier_lb = q1 - iqr * 3
    major_outlier_ub = q3 + iqr * 3
    is_outlier = (df.dropna() < outlier_lb) | (df.dropna() > outlier_ub)
    is_major_outlier = (df.dropna() < major_outlier_lb) | \
                       (df.dropna() > major_outlier_ub) 
    n_outliers = is_outlier.sum()
    n_major_outliers = is_major_outlier.sum()
    n_minor_outliers = n_outliers - n_major_outliers

    values = [mean, median, std, skew, kurt, min, argmin, max, argmax, q1, q3,
              iqr, outlier_lb, outlier_ub, major_outlier_lb, major_outlier_ub,
              n_obs, n_outliers, n_major_outliers, n_minor_outliers] 

    labels = ['mean', 'median', 'std', 'skew', 'kurt', 'min', 'argmin', 'max',
              'argmax', 'q1', 'q3', 'iqr', 'outlier_lb', 'outlier_ub',
              'major_outlier_lb', 'major_outlier_ub', 'n_obs', 'n_outliers',
              'n_major_outliers', 'n_minor_outliers']

    summary = pd.Series(values, labels)

    return summary
    

def remove_outliers(df, coeff=1.5):

    result = pd.DataFrame(index=df.index)
    columns = range(0, df.shape[1])

    for i in columns:
        column = df[[i]].dropna()
        q3, q1 = np.percentile(column, [75, 25])
        iqr = q3 - q1
        outlier_lb = q1 - iqr * coeff
        outlier_ub = q3 + iqr * coeff
        is_not_outlier = (column > outlier_lb) & (column < outlier_ub)
        result = result.join(column[is_not_outlier])

    return result.dropna()


# subset features to determine outliers
subset = data[['adj_volume', 'momentum', 'daily_returns']]

# generate indexes to create datasets
no_outliers_ix = remove_outliers(subset).index
no_major_outliers_ix = remove_outliers(subset, 3).index
with_outliers_ix = subset.dropna().index

# create master dataset
features = data[['adj_volume',
                 'momentum',
                 'momentum_by_volume',
                 'daily_returns']]

# build dataset based on previously generated indexes
no_outliers = features.loc[no_outliers_ix]
no_major_outliers = features.loc[no_major_outliers_ix]
with_outliers = features.loc[with_outliers_ix]

# receive array of dates, return respective day of week
# 0 for monday, 1 for tuesday, ... , 6 for sunday
def weekdays(dates): 
    weekdays = []
    for date in dates: weekdays.append(date.weekday())
    return np.array(weekdays)

# return true if date in array is monday, false otherwise
def mondays(dates):
    mondays = dates[weekdays(dates) == 0]
    return mondays

# substitute outliers above upper bound with upper bound value and outliers
# below lower bound with lower bound value
# the purpose of this is to not let wild outliers to screw the rest of the data
# by making it to take too small values 
def prune_outliers(df, coeff=1.5):
    result = pd.DataFrame(index=df.index)
    columns = range(0, df.shape[1])
    for i in columns:
        column = df[[i]].dropna()
        q3, q1 = np.percentile(column, [75, 25])
        iqr = q3 - q1
        outlier_lb = q1 - iqr * coeff
        outlier_ub = q3 + iqr * coeff
	is_below_lb = column < outlier_lb
	is_above_ub = column > outlier_ub
	# prune outliers by setting them equal to respectively lower or upper
	# bound 
	column[is_below_lb] = outlier_lb
	column[is_above_ub] = outlier_ub
     	result = result.join(column)
    return result.dropna()


# generate list of training and test sets
def generate_sets(features, window_size=90, step_size=6, prune_coeff=1.5):
    training_sets = []
    test_sets = []
    is_greater_than_window = mondays(features.index) > \
                             (mondays(features.index)[0] +
                              timedelta(days=window_size))
    mondays_range = mondays(features.index)[is_greater_than_window] 
    for monday in mondays_range:
        # slice training window
        train_window_start = monday - timedelta(days=window_size)
        train_window_end = monday - timedelta(days=1)
        train_window = features.loc[train_window_start:train_window_end]
        # slice test window
        test_window_end = monday + timedelta(days=step_size)
        test_window = features.loc[monday:test_window_end]
        # prune outliers in training window
        train_window = prune_outliers(train_window, prune_coeff)
	# prune outliers in test window
        # this can't be done by using prune_outliers function because this
        # function computes the function based on the dataset it is given
        # the problem with this is that - when it comes to outliers detection -
        # we would rather treat the test set as an extension of the training
        # set since at the time of decision making not all values of the test
        # set are known
        # therefore we substitute all values in test set greater than
        # max of pruned training set with max of pruned training set
        # itself instead of some computed upper bound on the  test set
        is_to_prune = test_window > train_window.max()
        # invert sign of is_to_prune to subset values to leave unchanged
	is_not_to_prune = abs(is_to_prune - 1)
        # create dataframe of same dimension as test window containing
        # pruned outliers and zeros everywhere else
        pruned_outliers = is_to_prune * train_window.max()
        # create dataframe of same dimension as test window containing
        # non outliers and zeros intead of outliers
	non_outliers = is_not_to_prune * test_window
        # obtain test window with pruned outliers by summing non_outliers and
        # pruned_outliers  
	test_window = non_outliers + pruned_outliers
        # normalize features
        train_window_values = minmax_scaler.fit_transform(train_window)
        train_window = pd.DataFrame(train_window_values,
                                    train_window.index,
                                    train_window.columns)
        # for the same reason explained above, we can't normalize the test set
        # data on their own, but rather as if they were part of the training set
        # for this reason, we implement the minmax scaler manually subtracting
        # min of training from each value of test_window and dividing by range
        # of training set (max - min)
        test_window_values = (test_window - train_window.min()) / \
                             (train_window.max() - train_window.min())
        test_window = pd.DataFrame(test_window_values,
                                   test_window.index,
                                   test_window.columns)
        training_sets.append(train_window)
        test_sets.append(test_window)
    return training_sets, test_sets


X_trains, X_tests = generate_sets(features.dropna())

# label based on sign the the return on the following day
labels = np.sign(features['daily_returns'].shift(-1))
# choose regressor
logreg = linear_model.LogisticRegression()
# fit regressor with 42nd X and y, just as an example to test 
logreg.fit(X_trains[42], labels[X_trains[42].index])
# subset labels for 42nd test set
y_true = labels[X_tests[42].index].values
# predict labels for 42nd test set
y_pred = logreg.predict(X_tests[42])
# compute PnL of the window by calculating y_pred (-1, 0 or 1) times the actual
# return of the given day
window_pnl = np.sum(y_pred * X_tests[42]['daily_returns'])
# compute accuracy, precision, recall and f1 measures
accuracy = metrics.accuracy_score(y_true, y_pred)
precision = metrics.precision_score(y_true, y_pred)
recall = metrics.recall_score(y_true, y_pred)
f1 = metrics.f1_score(y_true, y_pred)

# function takes as input a model, a list of training sets, a list of test sets
# and the labels
def run_model(model, X_trains, X_tests, ys):
    model = model
    accuracies = []
    accuracies_ix = []
    pnl = []
    for i in range(13, len(X_trains)):
        X_train = X_trains[i]
        X_test = X_tests[i]
        y = ys[X_train.index]
        model.fit(X_train, y)
        y_true = ys[X_test.index].values
        y_pred = model.predict(X_test)
        actual_returns = features['daily_returns'].loc[X_test.index]
        window_pnl = y_pred * actual_returns
	accuracy = metrics.accuracy_score(y_true, y_pred)
        accuracies.append(accuracy)
	accuracies_ix.append(X_test.index[0])
	pnl.append(window_pnl)
    accuracies = pd.DataFrame(accuracies, accuracies_ix, ['accuracy'])
    pnl = pd.DataFrame(pd.concat(pnl))
    pnl.columns = ['pnl']
#    summary = pd.DataFrame([accuracies.mean()[0], pnl.sum()[0]],
#			   columns=['summary'],
#			   index=['avg_accuracy', 'final_pnl'])
    return accuracies, pnl

# run logistic regression 
accuracy_lr, pnl_lr = run_model(linear_model.LogisticRegression(),
                                X_trains,
                                X_tests,
                                labels) 

# plot graph of cumulative profits with logistic regression model
plt.plot(np.cumsum(pnl_lr))


# run support vector machines
accuracy_svm, pnl_svm = run_model(svm.SVC(),
                                  X_trains,
                                  X_tests,
                                  labels) 

# plot graph of cumulative profits with logistic regression model
plt.plot(np.cumsum(pnl_svm))


# run nearest neighbor classifier
accuracy_knn, pnl_knn = run_model(neighbors.KNeighborsClassifier(),
                                  X_trains,
                                  X_tests,
                                  labels) 

# plot graph of cumulative profits with logistic regression model
plt.plot(np.cumsum(pnl_knn))


# compute ensemble pnl
pnl_ensemble = (pnl_lr + pnl_svm + pnl_knn) / 3

# plot graph of cumulative profits with ensemble model
plt.plot(np.cumsum(pnl_ensemble))

def evaluate(pnl):
    pnl = pnl.iloc[:, 0]
    final_pnl = pnl.sum()
    print final_pnl
    # avg annual return = avg daily return * number of trading days in a year
    avg_annual_return = pnl.mean() * 252
    # avg annual std = daily std * sqrt(number of trading days in a year)
    # we take the sqrt because
    # std = sqrt(variance)
    # annual variance = daily variance * 252
    # annual std = sqrt(annual variance)
    #            = sqrt(daily variance * 252)
    #            = sqrt(daily variance) * sqrt(252)
    #            = daily std * sqrt(252)
    annual_std = pnl.std() * np.sqrt(252)
    cum_ret = np.cumsum(pnl)
    drawdowns = np.maximum.accumulate(cum_ret) - cum_ret
    max_drawdown_valley = np.argmax(drawdowns)
    print max_drawdown_valley
    max_drawdown_peak = np.argmax(cum_ret[:max_drawdown_valley])
    max_drawdown = np.max(drawdowns)
    avg_drawdown = np.mean(drawdowns)
    median_drawdown = np.median(drawdowns)
    q3_drawdown = np.percentile(drawdowns, 75)
    # remember to compute recovery time from drawdowns
    # risk-adjusted return measures
    # all such measures are annualized here
    # information ratio = sharpe ratio without taking risk-free rate into account
    # using Narang's definition here, not Balch's - hence ignoring beta
    information_ratio = avg_annual_return / annual_std
    # sterling ratio = avg return / std of below average returns
    below_avg_returns = pnl[pnl < pnl.mean()]
    annual_below_avg_returns_std = below_avg_returns.std() * np.sqrt(252)
    sterling_ratio = avg_annual_return / annual_below_avg_returns_std
    # sterling_ratio = sterling_ratio.values[0]
    # adjusted sterling ratio = avg return / std of negative returns
    negative_returns = pnl[pnl < 0]
    annual_negative_returns_std = negative_returns.std() * np.sqrt(252)
    adjusted_sterling_ratio = avg_annual_return / annual_negative_returns_std
    # adjusted_sterling_ratio = adjusted_sterling_ratio.values[0] 
    calmar_ratio = avg_annual_return / max_drawdown
    # omega ratio = sum of all positive returns / sum of all negative returns
    positive_returns = pnl[pnl > 0]
    omega_ratio = positive_returns.sum() / np.abs(negative_returns.sum())
    # omega_ratio = omega_ratio.values[0]
    metrics_values = [final_pnl, avg_annual_return, annual_std, max_drawdown,
                      avg_drawdown, median_drawdown, information_ratio,
                      sterling_ratio, adjusted_sterling_ratio, calmar_ratio,
                      omega_ratio]
    metrics_labels = ['final_pnl', 'avg_annual_return', 'annual_std',
                      'max_drawdown', 'avg_drawdown', 'median_drawdown',
                      'information_ratio', 'sterling_ratio',
                      'adjusted_sterling_ratio', 'calmar_ratio', 'omega_ratio']
    metrics_df = pd.DataFrame(metrics_values, metrics_labels, ['value'])
    return metrics_df


features_norm_values = minmax_scaler.fit_transform(features.dropna())
features_norm_index = features.dropna().index
features_norm = pd.DataFrame(features_norm_values,
                             features_norm_index,
                             features.columns)



def generate_sets(features, window_size=90, step_size=6):
    training_sets = []
    test_sets = []
    is_greater_than_window = mondays(features.index) > \
                             (mondays(features.index)[0] +
                              timedelta(days=window_size))
    mondays_range = mondays(features.index)[is_greater_than_window] 
    for monday in mondays_range:
        # slice training window
        train_window_start = monday - timedelta(days=window_size)
        train_window_end = monday - timedelta(days=1)
        train_window = features.loc[train_window_start:train_window_end]
        # slice test window
        test_window_end = monday + timedelta(days=step_size)
        test_window = features.loc[monday:test_window_end]
        # prune outliers in training window
        train_window = prune_outliers(train_window)
	# prune outliers in test window
        # this can't be done by using prune_outliers function because this
        # function computes the function based on the dataset it is given
        # the problem with this is that - when it comes to outliers detection -
        # we would rather treat the test set as an extension of the training
        # set since at the time of decision making not all values of the test
        # set are known
        # therefore we substitute all values in test set greater than
        # max of pruned training set with max of pruned training set
        # itself instead of some computed upper bound on the  test set
        is_to_prune = test_window > train_window.max()
        pruned_outliers = is_to_prune * train_window.max() 
        test_window[is_to_prune] = pruned_outliers
        # normalize features
        train_window_values = minmax_scaler.fit_transform(train_window)
        train_window = pd.DataFrame(train_window_values,
                                    train_window.index,
                                    train_window.columns)
        # for the same reason explained above, we can't normalize the test set
        # data on their own, but rather as if they were part of the training set
        # for this reason, we implement the minmax scaler manually subtracting
        # min of training from each value of test_window and dividing by range
        # of training set (max - min)
        test_window_values = (test_window - train_window.min()) / \
                             (train_window.max() - train_window.min())
        test_window = pd.DataFrame(test_window_values,
                                   test_window.index,
                                   test_window.columns)
        training_sets.append(train_window)
        test_sets.append(test_window)
    return training_sets, test_sets


test_window = features.loc[X_tests[1201].index]
train_window = features.loc[X_trains[1201].index]
test_window.iloc[3,3] = 1
test_window.iloc[1,1] = 1
is_to_prune = test_window > train_window.max()
is_not_to_prune = abs(is_to_prune - 1)
pruned_outliers = is_to_prune * train_window.max()
non_outliers = is_not_to_prune * test_window
test_window = non_outliers + pruned_outliers

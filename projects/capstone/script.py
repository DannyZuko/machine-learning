import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


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

metrics = pd.DataFrame(metrics_values, metrics_labels, ['value'])

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

subset = data.loc[:, ['adj_volume',
                      'daily_returns',
                      'momentum',
                      'momentum_by_volume']] 

pd.scatter_matrix(subset, alpha = 0.3, figsize = (14,8), diagonal = 'kde')
# explain why excluding columns from scatter matrix

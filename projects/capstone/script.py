import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('WIKI-AAPL.csv', index_col=0)
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

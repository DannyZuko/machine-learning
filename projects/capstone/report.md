Capstone Project

Machine Learning Engineer Nanodegree

# Definition

## Project Overview

## Problem Statement

The problem we aim at solving is to find an optimal policy to buy and sell some given stock in order to make profits out of difference between buying price and selling price. We will try to find a solution to this problem through a reinforcement learning approach. In particular, we plan to use Q-learning. We will choose a set of relevant features, on the assumption that their combination might have some predictive power.

In a nutshell, we will be satisfied with the solution if the graph of cumulative profits over time has an upward trending slope and the ratio between the sum of all profits and the sum of all losses is at least greater than 2. This will mean that our model makes twice as much money than it loses, thus ensuring both profitability and low volatility.

## Metrics

* Graph of cumulative profits over time: this shows if the amount cumulative profits made by our agent are a positive amount, how smoothly profits are made with respect to the losses and the downside risk of the periods of negative performance

* Profits minus losses: how much the agent has made minus how much the agent has lost

* Average profit or loss per trade

* Standard deviation of profits and losses

* Percentage of winning trades over all trades (n. of winning trades + n. of losing trades)

* Average profit of winning trades

* Average loss of losing trades

* Drawdowns: amount of money lost from any cumulative peak in the profit curve

    * Worst peak-to-valley drawdown

    * Average drawdown

    * Median drawdown

    * 3rd quartile drawdown

* Recovery time from drawdowns: amount of time it takes to our agent to recover the amount of money previously lost

    * Worst peak-to-valley drawdown recovery time

    * Average drawdown recovery time

    * Median drawdown recovery time

    * 3rd quartile drawdown recovery time

* Risk-adjusted profit or loss measures:

    * Average profit or loss per trade divided by standard deviation of profits and losses

    * Average profit or loss per trade divided by standard deviation of losses only

    * Average profit or loss per trade divided by worst peak-to-valley drawdown

    * Sum of money made of winning trades divided by sum of money lost on losing trades

Our ideal agent:

* makes more money than it loses,

* shows a smooth upward trending profits curve,

* makes high average profits,

* makes low average losses,

* makes high percentage of winning trades,

* makes small drawdowns,

* has short recovery time from drawdowns

# Analysis

## Data Exploration

The dataset consists of financial time series of Apple Inc stock (ticker: AAPL) and comes from Quandl’s open source wiki ([https://www.quandl.com/data/WIKI/AAPL-Apple-Inc-AAPL-Prices-Dividends-Splits-and-Trading-Volume](https://www.quandl.com/data/WIKI/AAPL-Apple-Inc-AAPL-Prices-Dividends-Splits-and-Trading-Volume)).

Along with its daily timestamp as row index, each row contains the following inputs:

* Open: price at which the stock was traded when market opened

* High: highest price at which the stock was traded during the day

* Low: lowest price at which the stock was traded during the day

* Close: price at which the stock was traded when market closed

* Volume: number of stocks traded during the day

* Dividends: amount of money paid by Apple to its stockholders per share

* Splits: see note below

* Split/dividend adjusted open: see note below

* Split/dividend adjusted high: see note below

* Split/dividend adjusted low: see note below

* Split/dividend adjusted close: see note below

* Split/dividend adjusted volume: see note below

(write note explaining split/dividend adjusted price)

For an additional intuitive explanation of how this work, please watch the following video from Udacity’s Machine Learning for Trading course: [https://youtu.be/M2res0zhqjo](https://youtu.be/M2res0zhqjo)

The data per se - as it is given - offers little useful information until it has been preprocessed in order to extract relevant information out of the given data. We postpone documenting such preprocessing steps to the Methodology section.

The only useful information contained in the raw data is the adjusted volume. From now on, we will just refer to it as volume instead of adjusted volume in that it is the only data regarding volume that we are interested in.

The reason why we are interested in volume data is that our fundamental assumption is that price changes are driven by supply and demand and the number of stocks traded indicates whether the price changes are due to a handful of players or to a much larger crowd.

For example, let’s imagine a market where there are very few buyers and sellers. Then, the price will be determined by the supply and demand of those few players. On the contrary, if the crowd trading some market is very large, the equilibrium price will be the result of much wider negotiations. Therefore, we tend to assume that large volumes contribute to more reliable buying and selling signals. In any case, the size of the market - i.e. the number of stocks traded - does matter and we will try to use it to predict price movements, together with other human-engineered features.

Here are a few observations regarding the volume statistics:

* There are 9011 data points, one per trading day from 1980-12-12 to 2016-09-06

* The mean volume is about 91 mln and the median volume is about 63 mln

* The standard deviation of the volume is about 88 mln, which looks pretty high

* The skewness is above 3, which looks high too. We can also notice that the mean is much higher than the median. This suggests us that there are some large outliers on the right side of the distribution.

* The kurtosis is above 28, which is a very high reading. This indicates the tails of the distribution are quite fat.

* The minimum volume was around 250k on 1985-09-27, so during the 80s

* The maximum volume was around 1.9bn (!) on 2000-09-29, during the tech bubble

* There are 616 outliers and 191 major outliers. That is respectively 6.8% and 2.1% of the observations, which looks high. Also, given the outlier lower bound is negative, we infer that all outliers are on the right side of the distribution, since it is not possible the number of stocks traded on some given day is below zero. This tells us there were several days of extraordinary activity in the trading of Apple stocks and it is certainly something to keep in mind.

## Exploratory Visualization

As we can see from the box-and-whisker plot below there are many outliers and some of them are completely elsewhere with respect to the rest of the sample.  This confirms the impression we got in the Data Exploration subsection.

The plot changes substantially when we remove all major outliers from the picture, i.e. all outliers below Q1 - IQR * 3 and above Q3 + IQR * 3.

Eventually, the plot changes even more when we remove all outliers - both major and minor - from the picture, i.e. all outliers below Q1 - IQR * 1.5 and above Q3 + IQR * 1.5.

Such visualization gives us a visual intuition of what we observed about outliers in the Data Exploration subsection. The number of outliers is high and this will have to be taken into consideration when building the model.

Visualizing the histogram in order to get a visual intuition of the distribution of our volume data confirms what we observed in the Data Exploration subsection. The distribution is clearly positively skewed and it has a very long right tail. Also, it seems the distribution could be approximated by a lognormal distribution.

This is confirmed by the fact that if we do the same plot using the natural logarithm of the data we have a relatively nice bell curve. We will keep this in mind and see if it is of any help for the rest of the project.

## Algorithms and Techniques

## Benchmark

# Methodology

## Data Preprocessing

## Implementation

## Refinement

# Results

## Model Evaluation and Validation

## Justification

# Conclusion

## Free-form Visualization

## Reflection

## Improvement


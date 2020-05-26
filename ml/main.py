import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from commons.utils import Download


class SimpleLinearRegression:

    def __init__(self, is_download: bool = False, filename: str = 'FuelConsumption.csv'):
        self.url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv'
        if is_download:
            Download(url=self.url, filename=filename)
        self.df = pd.read_csv('download/FuelConsumption.csv')

    def data_exploration(self) -> None:
        print('[+] Show data head')
        print(self.df.head())
        print('[+] Summarize the data')
        print(self.df.describe())

    def plot_features(self) -> None:
        # cdf = self.df[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"]]
        cdf = self.df[self.df.columns.values]
        cdf.hist()
        plt.show()
        plt.close()

    def plot_linear(self, dependent: str = None, independent: str = None) -> None:
        plt.scatter(self.df[independent], self.df[dependent], color='blue')
        plt.xlabel(independent)
        plt.ylabel(dependent)
        plt.show()
        plt.close()

    def simple_regression_model(self):
        """Creating train and test dataset"""

        cdf = self.df[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"]]
        msk = np.random.rand(len(self.df)) < 0.8
        train = cdf[msk]
        test = cdf[~msk]

        # Training Model with Linear Regression
        regr = LinearRegression()
        train_x = np.asanyarray(train[['ENGINESIZE']])
        train_y = np.asanyarray(train[['CO2EMISSIONS']])
        regr.fit(train_x, train_y)

        # The coefficients
        print('Coefficient:', regr.coef_)
        print('Intercept:', regr.intercept_)

        plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
        plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], '-r')
        plt.xlabel('Engine Size')
        plt.ylabel('Emissions')
        plt.show()
        plt.close()

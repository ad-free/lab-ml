import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np


class SimpleLinearRegression:

    def __init__(self, is_download: bool = False):
        if is_download:
            self.download()
        self.df = pd.read_csv('FuelConsumption.csv')

    @staticmethod
    def download(path: str = None) -> bool:
        if not path:
            path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv'
        data = pd.read_csv(path)
        data.to_csv('FuelConsumption.csv')
        return True

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


if __name__ == '__main__':
    model = SimpleLinearRegression(is_download=True)
    # model.data_exploration()
    model.plot_features()
    model.plot_linear(dependent='CO2EMISSIONS', independent='ENGINESIZE')

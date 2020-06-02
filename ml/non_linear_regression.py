import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from commons.utils import Download


class NonLinearRegression:

    def __init__(self, is_download: bool = False, filename: str = 'china_gdp.csv') -> None:
        if is_download:
            Download(
                url='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/china_gdp.csv',
                filename=filename
            )
        self.df = pd.read_csv('download/' + filename)

    def summarize(self):
        print(self.df.head(10))
        print(self.df.describe())

    def plotting_dataset(self) -> None:
        plt.figure(figsize=(8, 5))
        x_data, y_data = (self.df['Year'].values, self.df['Value'].values)
        plt.plot(x_data, y_data, 'ro')
        plt.xlabel('GDP')
        plt.ylabel('Year')
        plt.show()
        plt.close()

        # Training model
        beta_1 = 0.10
        beta_2 = 1990.0

        # logistic function
        Y_pred = self.sigmoid(x_data, beta_1, beta_2)
        plt.plot(x_data, Y_pred * 15000000000000.)
        plt.plot(x_data, y_data, 'ro')
        plt.show()

        # normalize our data
        xdata = x_data / max(x_data)
        ydata = y_data / max(y_data)

        popt, pcov = curve_fit(self.sigmoid, xdata, ydata)
        # print('beta_1 = %f, beta_2 = %f' % (popt[0], popt[1]))
        x = np.linspace(1960, 2015, 55)
        x = x / max(x)
        plt.figure(figsize=(8, 5))
        y = self.sigmoid(x, *popt)
        plt.plot(xdata, ydata, 'ro', label='data')
        plt.plot(x, y, linewidth=3.0, label='fit')
        plt.legend(loc='best')
        plt.xlabel('Year')
        plt.ylabel('GDP')
        plt.show()
        plt.close()

        self.training_model(xdata, ydata)

    def training_model(self, xdata, ydata):
        msk = np.random.rand(len(self.df)) < 0.8
        train_x = xdata[msk]
        test_x = xdata[~msk]
        train_y = ydata[msk]
        test_y = ydata[~msk]

        # Building model
        popt, pcov = curve_fit(self.sigmoid, train_x, train_y)

        # predict using test set
        y_hat = self.sigmoid(test_x, *popt)

        # Evaluation
        print('Mean absolute error: %.2f' % np.mean(np.absolute(y_hat - test_y)))
        print('Residual sum of squares (MSE): %.2f' % np.mean((y_hat - test_y) ** 2))
        print('R2-score: %.2f' % r2_score(y_hat, test_y))

    def sigmoid(self, x, beta_1: float, beta_2: float):
        y = 1 / (1 + np.exp(-beta_1 * (x - beta_2)))
        return y

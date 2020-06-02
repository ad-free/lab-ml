import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from commons.utils import Download


class KNearestNeighBor:

    def __init__(self, is_download: bool = False, filename: str = 'teleCust1000t.csv') -> None:
        if is_download:
            Download(
                url='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/teleCust1000t.csv',
                filename=filename
            )
        self.df = pd.read_csv('download/' + filename)

    def summarize(self):
        print(self.df.head())
        print(self.df.describe())

    def plot_features(self):
        # self.df[self.df.columns.values].hist(bins=50)
        self.df.hist(column='income', bins=50)
        plt.show()
        plt.close()

    def k_nearest_neighbor(self):
        x = self.df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender',
                     'reside']].values
        y = self.df['custcat'].values

        x = StandardScaler().fit(x).transform(x.astype(float))

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
        print('Train set:', x_train.shape, y_train.shape)
        print('Test set:', x_test.shape, y_test.shape)

        k = 10
        mean_acc = np.zeros((k - 1))
        std_acc = np.zeros((k - 1))

        for n in range(1, k):
            # Training model and Predict
            neighbor = KNeighborsClassifier(n_neighbors=n).fit(x_train, y_train)
            y_hat = neighbor.predict(x_test)
            mean_acc[n - 1] = metrics.accuracy_score(y_test, y_hat)
            std_acc[n - 1] = np.std(y_hat == y_test) / np.sqrt(y_hat.shape[0])
            print(mean_acc)

        plt.plot(range(1, k), mean_acc, 'g')
        plt.fill_between(range(1, k), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
        plt.legend(('Accuracy ', '+/- 3xstd'))
        plt.ylabel('Accuracy ')
        plt.xlabel('Number of Nabors (K)')
        plt.tight_layout()
        plt.show()
        print('The best accuracy was with', mean_acc.max(), ' with k=', mean_acc.argmax() + 1)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class ModelDevelopment:
    def __init__(self, url: str = None, width: int = 12, height: int = 10):
        self.df = self.get_data(url)
        self.width = width
        self.height = height

    @staticmethod
    def get_data(url: str = None):
        if not url:
            url = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv"

        df = pd.read_csv(url)
        df.to_csv("data.csv")
        print(df.corr())
        return df

    @staticmethod
    def residiual_plot(self):
        plt.figure(figsize=(self.width, self.height))
        sns.residplot(self.df["highway-mpg"], self.df["price"])
        plt.show()
        plt.close()

    def distribution_plot(self, data):
        plt.figure(figsize=(self.width, self.height))

        ax1 = sns.distplot(
            self.df["price"], hist=False, color="r", label="Actual Values"
        )
        sns.distplot(data, hist=False, color="b", label="Fitted Values", ax=ax1)

        plt.title("Actual vs Fitted Values for Price")
        plt.xlabel("Price (in dollars)")
        plt.ylabel("Proportion of Cars")
        plt.show()
        plt.close()

    @staticmethod
    def polynomial_regression_plot(model, independent_variable, dependent_variable, name):
        x_new = np.linspace(15, 55, 100)
        y_new = model(x_new)

        plt.plot(independent_variable, dependent_variable, ".", x_new, y_new, "-")
        plt.title("Polynomial Fit with Matplotlib for Price ~ Length")
        ax = plt.gca()
        ax.set_facecolor((0.898, 0.898, 0.898))
        fig = plt.gcf()
        plt.xlabel(name)
        plt.ylabel("Price of Cars")

        plt.show()
        plt.close()

    def single_linear_regression(self):
        Z = self.df[["highway-mpg"]]
        lm = LinearRegression()
        lm.fit(Z, self.df["price"])
        Yhat = lm.predict(Z)

        self.distribution_plot(Yhat)

    def multiple_linear_regression(self):
        Z = self.df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
        lm = LinearRegression()
        lm.fit(Z, self.df["price"])  # training model
        Yhat = lm.predict(Z)
        self.distribution_plot(Yhat)

        print(Z.shape)

    def polynomial_regression(self):
        x = self.df["highway-mpg"]
        y = self.df["price"]

        f = np.polyfit(x, y, 3)
        p = np.poly1d(f)

        self.polynomial_regression_plot(p, x, y, "highway-mpg")

        print(f)

    def polynomial_features(self):
        Z = self.df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]

        pr = PolynomialFeatures(degree=2)
        Z_pr = pr.fit_transform(Z)

        print("The original data is of {samples} samples and {features}"
              .format(samples=Z.shape[0], features=Z.shape[1]))
        print("After the transformation, there {samples} samples and {features} features"
              .format(samples=Z_pr.shape[0], features=Z_pr.shape[1]))

    def pipeline(self):
        Z = self.df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
        y = self.df["price"]

        Input = [
            ('scale', StandardScaler()),
            ('polynomial', PolynomialFeatures(include_bias=False)),
            ('model', LinearRegression())
        ]

        pipe = Pipeline(Input)

        print(pipe.fit(Z, y))
        ypipe = pipe.predict(Z)
        print(ypipe[0:4])


if __name__ == "__main__":
    model = ModelDevelopment()
    # model.residiual_plot()
    # model.single_linear_regression()
    # model.multiple_linear_regression()
    # model.polynomial_regression()
    # model.polynomial_features()
    model.pipeline()

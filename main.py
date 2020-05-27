from ml.simple_linear_regression import SimpleLinearRegression

# from .data_science import DataScience

if __name__ == '__main__':
    simple = SimpleLinearRegression(is_download=True)
    simple.plot_features()

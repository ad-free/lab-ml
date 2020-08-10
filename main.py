from ml.simple_linear_regression import SimpleLinearRegression
import argparse


class CommandLine:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Commandline for Machine Learning')
        self.parser.add_argument('-slr', '--simple-linear-regression',
                                 dest='simple_linear',
                                 action='store_true',
                                 help='Simple linear regression')
        self.parser.add_argument('-mlr', '--mul-linear-regression',
                                 dest='mul_linear',
                                 action='store_true',
                                 help='Multiple Linear regression')
        self.parser.add_argument('-knr', '--k-nearest-neighbor',
                                 dest='k_nearest_neighbor',
                                 action='store_true',
                                 help='K nearest neighbor')
        self.parser.add_argument('-nlr', '--non-linear-regression',
                                 dest='non_linear',
                                 action='store_true',
                                 help='Non linear regression')
        self.parser.add_argument('-dt', '--decision-tree',
                                 dest='decision_tree',
                                 action='store_true',
                                 help='Decision Tree')
        self.parser.add_argument('--download',
                                 dest='is_download',
                                 action='store_true',
                                 help='Download data source')

        # Parse and processing
        self.options = self.parser.parse_args()
        self.processing(action=self.options)

    def processing(self, action):
        if action.simple_linear:
            self.simple_linear_regression(action.is_download)
        elif action.mul_linear:
            self.multiple_linear_regression(action.is_download)

    @staticmethod
    def simple_linear_regression(is_download) -> None:
        slr = SimpleLinearRegression(is_download)
        slr.simple_regression_model()

    @staticmethod
    def multiple_linear_regression(is_download):
        pass

    def non_linear_regression(self):
        pass

    def k_nearest_neighbor(self):
        pass

    def decision_tree(self):
        pass


if __name__ == '__main__':
    CommandLine()

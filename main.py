# from ml.simple_linear_regression import SimpleLinearRegression
# from ml.non_linear_regression import NonLinearRegression
from ml.k_nearest_neighbor import KNearestNeighBor
from ml.decision_tree import DecisionTree

# from .data_science import DataScience

if __name__ == '__main__':
    decision_tree = DecisionTree(is_download=True)
    decision_tree.decision_tree()

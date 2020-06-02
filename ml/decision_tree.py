import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pydotplus
from commons.utils import Download
from sklearn import tree
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO


class DecisionTree:

    def __init__(self, is_download: bool = False, filename: str = 'drug200.csv') -> None:
        if is_download:
            Download(
                url='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv',
                filename=filename
            )
        self.df = pd.read_csv('download/' + filename, delimiter=',')

    def decision_tree(self):
        data = self.processing_data()
        predict_data = self.df['Drug']
        x_train, x_test, y_train, y_test = train_test_split(data, predict_data, test_size=0.3, random_state=3)
        drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
        drugTree.fit(x_train, y_train)
        predTree = drugTree.predict(x_test)

        dot_data = StringIO()
        filename = "drugtree.png"
        featureNames = self.df.columns[0:5]
        targetNames = self.df['Drug'].unique().tolist()
        out = tree.export_graphviz(drugTree,
                                   feature_names=featureNames, out_file=dot_data,
                                   class_names=np.unique(y_train), filled=True, special_characters=True,
                                   rotate=False)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png(filename)
        img = mpimg.imread(filename)
        plt.figure(figsize=(100, 200))
        plt.imshow(img, interpolation='nearest')
        plt.show()

    def processing_data(self):
        # Remove the column containing the target name since it doesn't contain numeric values
        data = self.df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
        le_sex = preprocessing.LabelEncoder()
        le_sex.fit(['F', 'M'])
        data[:, 1] = le_sex.transform(data[:, 1])

        le_BP = preprocessing.LabelEncoder()
        le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
        data[:, 2] = le_BP.transform(data[:, 2])

        le_Chol = preprocessing.LabelEncoder()
        le_Chol.fit(['NORMAL', 'HIGH'])
        data[:, 3] = le_Chol.transform(data[:, 3])

        return data

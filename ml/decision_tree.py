import pandas as pd
import numpy as np
import graphviz
from commons.utils import Download
from sklearn import tree
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


class DecisionTree:

    def __init__(self, is_download: bool = False, filename: str = 'drug200.csv') -> None:
        if is_download:
            Download(
                url='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs'
                    '/drug200.csv',
                filename=filename
            )
        self.df = pd.read_csv('download/' + filename, delimiter=',')

    def decision_tree(self) -> None:
        data = self.processing_data()
        predict_data = self.df['Drug']
        x_train, x_test, y_train, y_test = train_test_split(data, predict_data, test_size=0.3, random_state=3)
        drug_tree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
        drug_tree.fit(x_train, y_train)

        feature_names = self.df.columns[0:5]
        dot_data = tree.export_graphviz(drug_tree,
                                        feature_names=feature_names, out_file=None,
                                        class_names=np.unique(y_train), filled=True, special_characters=True,
                                        rotate=False)
        graph = graphviz.Source(dot_data)
        graph.render('iris')

    def processing_data(self):
        """Remove the column containing the target name since it doesn't contain numeric values
        :return:
        """

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

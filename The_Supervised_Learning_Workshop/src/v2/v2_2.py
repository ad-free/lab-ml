import json
import pandas as pd
import numpy as np
import missingno as msno
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns


class DestributionOfValue(object):
    def __init__(self):

        with open("../../types/v2/dtypes.json", "r") as types:
            self.dtype = json.load(types)

        self.data = pd.read_csv(
            r"../../datasets/v2/earthquake_data.csv", dtype=self.dtype
        )

        self.description_features = [
            "injuries_description",
            "damage_description",
            "total_injuries_description",
            "total_damage_description",
        ]

        imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="NA")

        self.data[self.description_features] = imp.fit_transform(
            self.data[self.description_features]
        )

        self.damage_description()

    def damage_description(self):
        """
        0 = NONE
        1 = LIMITED (roughly corresponding to less than $1 million)
        2 = MODERATE (~$1 to $5 million)
        3 = SEVERE (~>$5 to $24 million)
        4 = EXTREME (~$25 million or more)
        """

        damage_description_counts = self.data.damage_description.value_counts()
        damage_description_counts = damage_description_counts.sort_index()

        fig, ax = plt.subplots(figsize=(10, 10))
        slices = ax.pie(
            damage_description_counts,
            labels=damage_description_counts.index,
            colors=["white"],
            wedgeprops={"edgecolor": "black"},
        )

        patches = slices[0]
        hatches = ["/", "\\", "|", "-", "+", "x", "o", "O", "\.", "*"]

        for patch in range(len(patches)):
            patches[patch].set_hatch(hatches[patch])

        plt.title("Pie chart showing counts for\ndamage_description categories")
        plt.show()


if __name__ == "__main__":
    DestributionOfValue()

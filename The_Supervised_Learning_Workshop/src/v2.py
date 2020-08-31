import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from missingno import heatmap, matrix


class StatisticsAndMissingValues(object):
    def __init__(self, *args, **kwargs):
        self.data = pd.read_csv(r"../datasets/v2/house_prices.csv")

        # * Delete the columns having more than 80% of values missing.
        self.data = self.missing_values(self.data)

        # * Replace null values in the FireplaceQu column with NA value
        self.data["FireplaceQu"] = self.data["FireplaceQu"].fillna("NA")

    @staticmethod
    def summary_statistics(df: DataFrame) -> None:
        print(df.info())
        print("*" * 100)
        print(df.describe().T)

    def missing_values(self, df: DataFrame) -> DataFrame:
        # * Find columns have null values
        mask = df.isnull()
        total = mask.sum()
        percent = 100 * mask.mean()

        # * Get null columns and convert it to list
        nullable_collumns = df.columns[mask.any()].tolist()
        self._plot(df=df, columns=nullable_collumns, is_heatmap=True)

        # * Find missing value in data
        missing_data = pd.concat(
            [total, percent],
            axis=1,
            join="outer",
            keys=["count_missing", "perc_missing"],
        )
        missing_data.sort_values(by="perc_missing", ascending=False, inplace=True)
        missing_data[missing_data.count_missing > 0]

        return df.loc[:, missing_data[missing_data.perc_missing < 80].index]

    @staticmethod
    def _plot(df: DataFrame, columns: list = None, is_heatmap: bool = False) -> bool:
        if columns:
            matrix(df[columns].sample(500)) if not is_heatmap else heatmap(
                df[columns], vmin=-0.1, figsize=(18, 18)
            )
            plt.show()
            return True

        return False


if __name__ == "__main__":
    StatisticsAndMissingValues()


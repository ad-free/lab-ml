import pandas as pd


class Download:
    """ Download and prepare data """

    def __init__(self, url: str, filename: str = None):
        self.download(url, filename)

    @staticmethod
    def download(path: str, filename: str = None) -> bool:
        if not filename:
            filename = 'default.csv'
        data = pd.read_csv(path)
        data.to_csv('download/' + filename)
        return True

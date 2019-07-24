import pandas as pd

class LoadData:

    def __init__(self, path):

        self.path = path

    def get_data(self):
        try:
            return pd.read_csv(self.path)
        except:
            return None
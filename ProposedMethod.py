from DataLoder import LoadData
import numpy as np
import pandas as pd

class Cluster:

    def __init__(self, df, fc):
        self.__df = df # df is DataFrame
        self.__fc = fc # fc is feature in df that represent class
        self.__v = self.__center()
        self.__r = self.__radias()

    def size(self):
        try:
            shape = self.__df.shape
            return shape[0], shape[1]
        except:
            return 0, 0

    def get_data(self):
        return self.__df.values

    def sub_data(self,feature_list):
        return self.__df[feature_list].values

    def get_feature(self):
        return self.__df.columns.tolist()

    def number_of_class(self):
        data = self.__df.groupby(self.__fc).nunique()
        return (data.shape)[0]

    def __center(self):
        row , col = self.size()

        feature_list = self.get_feature()
        feature_list.remove(self.__fc)
        c = 1/row
        return c * sum(self.sub_data(feature_list))

    def __radias(self):
        row, col = self.size()

        feature_list = self.get_feature()
        feature_list.remove(self.__fc)

        if row <=0:
            return 0
        c = 1 / row

        v_star = self.__center()
        dist = (v_star - self.sub_data(feature_list)) ** 2
        dist = np.sum(dist)
        dist = np.sqrt(dist)

        return c * dist

    def freedom_degree(self, cluster):
        ci = self.__center()
        cj = cluster.__center()

        ri = self.__radias()
        rj = cluster.__radias()

        dist = (ci - cj) ** 2
        dist = np.sum(dist)
        dist = np.sqrt(dist)

        return dist - (ri + rj)

    def p(self):
        print(self.number_of_class())


class LW_index:

    def __init__(self, c1, c2):
        self.__ci = c1
        self.__cj = c2

    def lw(self):
        pass


def main():
    # read data
    loder = LoadData('/home/jahanbakhsh/PycharmProjects/ML/Featuer_Selection/dataset/ionosphere.csv')
    data_set = loder.get_data()

    c = Cluster(df=data_set, fc='class')
    c.p()

main()
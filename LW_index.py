import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from DataLoder import LoadData
from random import choice
from sklearn.neighbors.nearest_centroid import NearestCentroid


class DataSet:
    def __init__(self, da):
        self._data = da

    def get_data(self):
        return self._data.values

    def get_feature(self):
        return self._data.columns.tolist()

    def sub_data(self,feature_list):
        return self._data[feature_list]


    def get_feature_value(self, feature_list):
        return self._data[feature_list].values

    def sum(self,feature_list):
        return sum(self.get_feature_value(feature_list))

    def get_class_data(self, class_label, class_featuer):
        return self._data[self._data[class_label ]== class_featuer]

class LW_index:

    def __init__(self, ds):
        self._data_set =ds

    @staticmethod
    def v_star(data):
        # self._data_set.get_feature_value(feature_list)
        k = len(data)
        if k <=0:
            return 0
        else:
            c = 1/k
            return c *sum(data)

    @staticmethod
    def r_star( data):
        # self._data_set.get_feature_value(feature_list)
        k = len(data)
        if k <=0:
            return 0
        else:
            v_star = LW_index.v_star(data)
            c = 1/k
            dist = (v_star - data)**2
            dist = np.sum(dist)
            dist = np.sqrt(dist)

            return c * dist

    def freedom_degree(self,df_fi, df_fj):
        fi = DataSet(df_fi).get_data()
        fj = DataSet(df_fj).get_data()

        v_fi = self.v_star(fi)
        v_fj = self.v_star(fj)
        r_fi = self.r_star(fi)
        r_fj = self.r_star(fj)

        dist = (v_fi-v_fj)**2
        dist = np.sum(dist)
        dist = np.sqrt(dist)

        return dist - (r_fi + r_fj)

    def fc(self, fi, fj, class_list, class_label):
        LW=[]
        M = len(class_list)

        for cls in class_list:
            dfi = DataSet(fi)
            dfi = dfi.get_class_data(class_label, cls)
            dfi=dfi.drop(columns=[class_label])

            dfj = DataSet(fj)
            dfj = dfj.get_class_data(class_label,cls)
            dfj=dfj.drop(columns=[class_label])

            LW.append(self.freedom_degree(dfi, dfj))
        return 1/M * min(LW)

    def lw_index(self, fi, fj, featuer_set, class_label):
        fo = featuer_set
        fs = []

        while len(fo)==0:
            ft = list(fo)
            while len(ft)==0:
                fd = [ft[0]]
                fc =fs+fd
                ft.remove(fd[0])

                temp = list(fo)
                for k in fc:
                    temp.remove(k)
                lw = self.fc()

def main():
    # read data
    loder = LoadData('./dataset/ionosphere.csv')
    data_set = loder.get_data()

    d1 = data_set.sample(n=35)
    d2 = data_set.sample(n=55)

    data_set = DataSet(data_set)
    lw = LW_index(data_set)
    lw =lw.fc(d1, d2, ['g', 'b'], 'class')
    print(lw)

main()
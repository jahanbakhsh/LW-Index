import numpy as np
import pandas as pd

class DataSet:
    def __init__(self, data_frame, class_column):
        self.__data_frame = data_frame
        self.__class_column =class_column

    @property
    def data_set(self):
        return self.__data_frame

    @property
    def data(self):
        return self.__data_frame.values

    def get_data(self,features):
        return self.__data_frame[features]
    @property
    def feature(self):
        return self.__data_frame.columns.tolist()

    @property
    def class_column(self):
        return self.__class_column

    def sample(self,frac,random_state):
        return self.__data_frame.sample(frac=frac,random_state=random_state )

    def drop(self,index):
        return self.__data_frame.drop(index=index)

class Cluster:

    def __init__(self, df, fc):
        self.__df = df # df is DataFrame
        self.__fc = fc # fc is feature in df that represent class
        self.__v = self.__center()
        self.__r = self.__radias()

    @property
    def fc(self):
        return self.__fc

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

    def class_list(self):
        data_set = self.__df
        cls = self.__fc
        data = data_set.groupby(cls).indices.keys()
        cls_num = len(data)

        return list(data)


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

    def get_class_data(self, cls):
        df = self.__df
        fc = self.__fc
        data = []
        for c in cls:
            data.append(df[df[fc] == c])
        return pd.concat(data)


class LW_index:

    def __init__(self, c1, c2):
        self.__ci = c1
        self.__cj = c2


    def lw(self):
        ci = self.__ci
        cj = self.__cj


        ci_class = ci.class_list()
        cj_class = cj.class_list()

        sum =0

        for clsi in ci_class:
            cj_cls = list(cj_class)
            cj_cls.remove(clsi)

            new_ci = Cluster(df=ci.get_class_data([clsi]), fc=ci.fc)
            fd = []
            for clsj in cj_cls:
                new_cj = Cluster(df=cj.get_class_data([clsj]), fc=cj.fc)
                fd.append(new_ci.freedom_degree(new_cj))
                del new_cj
            sum += min(fd)
            del fd
        return sum

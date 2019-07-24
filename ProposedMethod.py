from DataLoder import LoadData
import numpy as np
import pandas as pd

class Cluster:

    def __init__(self, df, fc):
        self.__df = df # df is DataFrame
        self.__fc = fc # fc is feature in df that represent class
        self.__v = self.__center()
        self.__r = self.__radias()
        print('')

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
    def p(self, c):
        print(self.get_class_data(c))


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

            new_ci = Cluster(df=ci.get_class_data(clsi), fc=ci.fc)
            fd = []
            for clsj in cj_cls:
                new_cj = Cluster(df=cj.get_class_data(clsj), fc=cj.fc)
                fd.append(new_ci.freedom_degree(new_cj))
                del new_cj
            sum += min(fd)
            del fd
        return sum


def get_candidate_feature(featuer_list):
    if len(featuer_list) >0:
        f = featuer_list[0]
        featuer_list.remove(f)
        return f, featuer_list
    else:
        return None, featuer_list

def remain_featuer(orig_f, sub_f):
    new_f =[]
    for f in orig_f:
        if f not in sub_f:
            new_f.append(f)
    return new_f


def main():
    # read data
    loder = LoadData('/home/jahanbakhsh/PycharmProjects/Featuer_Selection/dataset/ionosphere.csv')
    data_set = loder.get_data()

    cls_featuer = 'class'
    clu = Cluster(df=data_set, fc=cls_featuer)

    original_feature = clu.get_feature()
    orig_f = list(original_feature)
    original_feature.remove(cls_featuer)
    selected_feature =[]
    ft =[]
    lw_list =[]
    lw_scor =[]
    t =0
    while ([] != original_feature) or t <100:
        temp_feature = list(original_feature)

        while []!=temp_feature:
            fc , temp_feature = get_candidate_feature(temp_feature)
            candidate_list = list(selected_feature)
            candidate_list.append(fc)

            cj = remain_featuer(orig_f,candidate_list)
            cj.append(cls_featuer)

            ci = list(candidate_list)
            ci.append(cls_featuer)

            cj = data_set[cj]
            ci = data_set[ci]

            cj=Cluster(cj, cls_featuer)
            ci=Cluster(ci, cls_featuer)

            lw = LW_index(ci, cj)
            lw_scor.append(lw.lw())
            lw_list.append(candidate_list)

        index = lw_scor.index(max(lw_scor))
        selected_feature.extend(lw_list[index])

        for f in lw_list[index]:
            original_feature.remove(f)
        t +=1


if __name__ == '__main__':
    main()
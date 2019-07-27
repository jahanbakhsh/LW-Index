from sklearn import svm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
class Expriment:

    def __init__(self,data_set, experiment_feature, cls_feature,expriment_name,result_path,cls_method='svm'):
        self.__train_set, self.__test_set = train_test_split(data_set.data_set, test_size=0.2)
        self.__experiment_feature = experiment_feature
        self.__cls_feature=cls_feature
        self.__result = {}
        self.__expriment_name = expriment_name
        self.__result_path = result_path
        self.__cls_method = cls_method

    def __remove_missing_values(self, data, missing_values='nan',strategy='mean',axis=0):
        data=data.fillna(data.mean())
        return data

    def __train(self, feature_list):
        train_set = self.__train_set
        train_set = train_set[feature_list]
        train_set = self.__remove_missing_values(data=train_set)
        target = train_set[self.__cls_feature]

        if self.__cls_feature in feature_list:
            feature_list.remove(self.__cls_feature)

        train_set=train_set[feature_list]
        data = train_set.values
        if self.__cls_method =='svm':
            classifier = svm.SVC(gamma='scale')
        else:
            classifier = NearestCentroid()
        classifier.fit(data, target)

        return classifier


    def __test(self,feature_list, classifier):
        test_set = self.__test_set
        test_set = test_set[feature_list]
        test_set = self.__remove_missing_values(data=test_set)

        target = test_set[self.__cls_feature]
        if self.__cls_feature in feature_list:
            feature_list.remove(self.__cls_feature)

        test_set = test_set[feature_list]
        data = test_set.values

        predictions = classifier.predict(data)

        return accuracy_score(target, predictions, normalize=False)/len(data)

    @property
    def result(self):
        return self.__result

    def __add_result(self, key, val):
        self.__result[key]=val

    def __figure_result(self):
        if len(self.result)>0:
            plt.plot(list(self.result.keys()), list(self.result.values()))
            plt.ylabel('Accuracy')
            plt.xlabel('Number of feature\n {} Dataset'.format(self.__expriment_name))
            plt.savefig('{}/{}_{}.png'.format(self.__result_path,self.__expriment_name,self.__cls_method), bbox_inches='tight')
            plt.clf()

    def run(self):
        items = self.__experiment_feature.items()
        for key, val in items:
            val.append(self.__cls_feature)
            svm_classifier = self.__train(list(val))
            a = self.__test(val,svm_classifier)

            self.__add_result(key=key, val=a)

        self.__figure_result()



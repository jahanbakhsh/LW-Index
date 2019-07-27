from DataLoder import LoadData
from LW_index import DataSet, Cluster, LW_index
import os
from Experiments import Expriment
from pandas.api.types import is_string_dtype



PATH_BASE= os.path.dirname(__file__)
PATH_RESULT = os.path.join(PATH_BASE, 'result')

DATA_SET = ['lungcancer',  'ionosphere','sonar','soybean']
CLS_METHOD =['svm', 'cbc']
# CLS_METHOD = 'svm'
# CLS_METHOD = 'cbc'

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


def main(data_set, cls_featuer, threshold):

    experiment_feature = {}

    original_feature = data_set.feature
    orig_f = list(original_feature)
    original_feature.remove(cls_featuer)
    orig_f.remove(cls_featuer)

    selected_feature =[]
    lw_list =[]
    lw_scor =[]
    t =0
    while ([] != original_feature) and t < threshold:
        temp_feature = list(original_feature)

        while []!=temp_feature:
            fc , temp_feature = get_candidate_feature(temp_feature)
            candidate_list = list(selected_feature)
            candidate_list.append(fc)

            ci = list(candidate_list)
            ci.append(cls_featuer)

            cj = list(candidate_list)
            cj.append(cls_featuer)

            cj = data_set.get_data(cj)
            ci = data_set.get_data(ci)

            cj=Cluster(cj, cls_featuer)
            ci=Cluster(ci, cls_featuer)

            lw = LW_index(ci, cj)
            lw_scor.append(lw.lw())
            lw_list.append(candidate_list)

        index = lw_scor.index(max(lw_scor))
        selected_feature=list(lw_list[index])


        for f in lw_list[index]:
            if f in original_feature:
                original_feature.remove(f)
                orig_f.remove(f)

        t +=1
        experiment_feature[t]= list(lw_list[index])

        lw_list = list([])
        lw_scor = list([])

    return experiment_feature

if __name__ == '__main__':


    for method in CLS_METHOD:
        for data_sent_name in DATA_SET:
            PATH_DATA = os.path.join(PATH_BASE,'dataset/{}.csv'.format(data_sent_name))

            print('start load {} dataset ...'.format(data_sent_name))

            loader = LoadData(PATH_DATA)
            cls_featuer = 'class'

            print('loading of {} dataset complated'.format(data_sent_name))

            print('start to cleaning {} dataset ... '.format(data_sent_name))
            data_set = loader.get_data()
            if is_string_dtype(data_set[cls_featuer]):
                class_value = data_set[cls_featuer].unique()
                class_value = list(class_value)

                vals = list(range(0, len(class_value)))
                dic = dict(zip(class_value, vals))
                data_set[cls_featuer] = data_set[cls_featuer].map(dic)

            data_set = DataSet(data_set, cls_featuer)
            print('cleaning {} dataset complated'.format(data_sent_name))
            THRESHOLD = len(data_set.feature)

            print('start to featuer subset selection in {} dataset ...'.format(data_sent_name))
            experiment_feature = main(data_set=data_set, cls_featuer=cls_featuer,
                                     threshold=THRESHOLD)

            print(experiment_feature)

            print('all subset extracted')
            print('start to compute resule of {} method on {} dataset'.format(method,data_sent_name))
            epr = Expriment(data_set=data_set,
                            experiment_feature=experiment_feature,
                            cls_feature=cls_featuer, expriment_name=data_sent_name,
                            result_path = PATH_RESULT,cls_method=method)
            epr.run()
            print('result saved')
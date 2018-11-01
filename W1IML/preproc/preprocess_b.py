import numpy as np
from sklearn import preprocessing as prp

class Preprocess_b:
    def preprocess_methodb(self, data1):
        # Preprocessing the database
        pre_pro = prp.LabelEncoder()  # Label Encoding
        pre_pro2 = prp.OneHotEncoder(sparse=False)  # One hot encoding

        # Preprocessing: Features may have different ranges
        scaler = prp.MinMaxScaler()
        variable = False
        features_to_delete = []
        n_features = data1.shape[1]
        print('\033[1m'+'Preprocessing:'+'\033[0m')
        for i in range(0, n_features):
            if not isinstance(data1[0][i], np.float):
                print('Feature ' + str(i) + ' has been encoded as one hot')
                data1[:, i] = pre_pro.fit_transform(data1[:, i])
                data = pre_pro2.fit_transform(data1[:, i].reshape(len(data1[:, i]), 1))
                data1 = np.concatenate((data1, data), axis=1)
                features_to_delete.append(i)
                variable = True

            else:
                print('Feature ' + str(i) + ' has NOT been encoded as one hot')
                data = data1[:, i].reshape(len(data1[:, i]), 1)
                data = np.float64(data)
                inds = np.where(np.isnan(data))
                mean_data = np.nanmean(data, axis=0)
                data[inds] = np.take(mean_data, inds[1])
                res = scaler.fit_transform(data)
                data1[:, i] = res.reshape(len(res), )

        if variable:
            data1 = np.concatenate((data1, data1[:, n_features].reshape(len(data1[:, i]), 1)), axis=1)
            data1 = np.delete(data1, features_to_delete, axis=1)
            #data1 = np.delete(data1, 0, axis=1)

        return data1
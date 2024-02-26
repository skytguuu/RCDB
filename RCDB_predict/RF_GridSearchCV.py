import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import random
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('model255_simi_input_dataclean.txt', sep = '\t')
folds = []
molecular = np.unique(data['CID'])
random.seed(2)
random.shuffle(molecular)
for i in range(9):
    folds.append(molecular[i*25: (i+1)*25])
folds.append(molecular[9*25: ])
Train_data_folds = []
Test_data_folds = []
for i in range(10):
    test = (data['CID'] == 'LALA') # all False
    for j in folds[i]:
        test += (data['CID'] == j)
    Train_data_folds.append(data[~test])
    Test_data_folds.append(data[test])

features = np.zeros((2059, 10))
AUC = []
Accuracy = []
Recall = []
F1_score = []
Precesion = []
for i in range(1):
    print('Train data shape:',Train_data_folds[i].shape)
    print('TestA data shape:',Test_data_folds[i].shape)
# Train_data.head()
# Train_data.info()
# Train_data.describe()
    Y_train = Train_data_folds[i]['y']
    X_train = Train_data_folds[i].drop(['y', 'GOID', 'CID', 'Pairs'], axis=1)
    Y_test = Test_data_folds[i]['y']
    X_test = Test_data_folds[i].drop(['y', 'GOID', 'CID', 'Pairs'], axis=1)
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_test, label=Y_test)

    print('*' * 25, 'begin train ', i, '*' * 25)
    rf = RandomForestClassifier()
    parameters = {'n_estimators': range(50, 80, 10), 'max_depth': range(10, 18, 2),
                  'min_samples_leaf': range(6, 16, 2), 'max_features': range(1, 6, 2)}
    grid_rfc = GridSearchCV(rf, parameters)
    grid_rfc.fit(X_train, Y_train)
    print(grid_rfc.best_params_, grid_rfc.best_score_)

    rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=16, max_features=3, max_leaf_nodes=None,
            min_samples_leaf=14, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=70, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
    rf.fit(X_train, Y_train)
    print(rf.get_params(deep=True))

    # 对测试集进行预测
    ypred = rf.predict(X_test)
    Y_pred = (ypred > 0.49) * 1
    AUC.append(metrics.roc_auc_score(Y_test, ypred))
    Accuracy.append(metrics.accuracy_score(Y_test, Y_pred))
    Recall.append(metrics.recall_score(Y_test, Y_pred))
    F1_score.append(metrics.f1_score(Y_test, Y_pred))
    Precesion.append(metrics.precision_score(Y_test, Y_pred))

print('AUC: %.4f (%.4f)' % (np.mean(AUC), np.std(AUC)))
print('Accuracy: %.4f (%.4f)' % (np.mean(Accuracy), np.std(Accuracy)))
print('Recall: %.4f (%.4f)' % (np.mean(Recall), np.std(Recall)))
print('F1_score: %.4f (%.4f)' % (np.mean(F1_score), np.std(F1_score)))
print('Precesion: %.4f (%.4f)' % (np.mean(Precesion), np.std(Precesion)))


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import metrics
import random


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
    test = (data['CID'] == 'LALA')
    for j in folds[i]:
        test += (data['CID'] == j)
    Train_data_folds.append(data[~test])
    Test_data_folds.append(data[test])

AUC = []
Accuracy = []
Recall = []
F1_score = []
Precesion = []
for i in range(10):
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
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': [ 'logloss','auc'],
        'max_depth': 2,
        'subsample': 0.95,
        'min_child_weight': 18,
        'colsample_bytree': 0.95,
        'eta': 0.05,
        'gamma': 0,
        'lambda': 1.5,
        'silent': 1,
        'seed': 1000,
        'nthread': 4,
    }
    print('*' * 25, 'begin train ', i, '*' * 25)
    model = xgb.train(params,
                      dtrain=dtrain,
                      verbose_eval=True,
                      evals=[(dtrain, "train"), (dtest, "valid")],
                      early_stopping_rounds=10,
                      num_boost_round=1000
                      )

    # 对测试集进行预测
    dtest = xgb.DMatrix(X_test)
    ypred = model.predict(dtest)
    Y_pred = (ypred > 0.49) * 1
    AUC.append(metrics.roc_auc_score(Y_test, ypred))
    Accuracy.append(metrics.accuracy_score(Y_test, Y_pred))
    Recall.append(metrics.recall_score(Y_test, Y_pred))
    F1_score.append(metrics.f1_score(Y_test, Y_pred))
    Precesion.append(metrics.precision_score(Y_test, Y_pred))

    # 此处自定义特征重要性指标
    importance_eval_list = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
    for j, importance_type in enumerate(importance_eval_list):
        feat_importance = model.get_score(importance_type=importance_type)
        feat_importance = pd.DataFrame.from_dict(feat_importance, orient='index')
        feat_importance.columns = [importance_type]
        if j == 0:
            df_temp = feat_importance
        else:
            df_temp = pd.merge(df_temp, feat_importance, how='outer', left_index=True, right_index=True)

    model_suffix0 = 'fold' + str(i)
    feat_importance_name = 'model255_simi_feat_importance_%s.csv' % (model_suffix0)
    df_temp.to_csv(feat_importance_name, index=True)


print('AUC: %.4f (%.4f)' % (np.mean(AUC), np.std(AUC)))
print('Accuracy: %.4f (%.4f)' % (np.mean(Accuracy), np.std(Accuracy)))
print('Recall: %.4f (%.4f)' % (np.mean(Recall), np.std(Recall)))
print('F1_score: %.4f (%.4f)' % (np.mean(F1_score), np.std(F1_score)))
print('Precesion: %.4f (%.4f)' % (np.mean(Precesion), np.std(Precesion)))



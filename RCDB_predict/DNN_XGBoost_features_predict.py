
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import random


class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        # self.linear1 = torch.nn.Linear(131, 65)
        self.linear1 = torch.nn.Linear(184, 92)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(92, 2)
        # self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        # x = self.softmax(x)
        return x

# data = pd.read_csv('model255_simi_input_dataclean_RF_features.txt', sep = '\t')
# data = pd.read_csv('model255_simi_input_dataclean_RF_features_1170.txt', sep = '\t')
data = pd.read_csv('model255_simi_input_dataclean_features.txt', sep = '\t')

model = TinyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.02)
loss_fn = torch.nn.CrossEntropyLoss()
print('Train data shape:',data.shape)
print('TestA data shape:',data.shape)
# Train_data.head()
# Train_data.info()
# Train_data.describe()
Y_train = data['y']
X_train = data.drop(['y', 'GOID', 'CID', 'Pairs'], axis=1)
Y_test = data['y']
X_test = data.drop(['y', 'GOID', 'CID', 'Pairs'], axis=1)
X_train = torch.tensor(X_train.values, dtype=torch.float)
Y_train = torch.tensor(Y_train.values, dtype=torch.long)
X_test = torch.tensor(X_test.values, dtype=torch.float)
Y_test = torch.tensor(Y_test.values, dtype=torch.long)

train_loss = []
val_loss = []
AUC = []
Accuracy = []
Recall = []
F1_score = []
Precesion = []
for t in range(150):
    model.train(True)
    ypred = model(X_train)     # 喂给 net 训练数据 x, 输出分析值
    loss = loss_fn(ypred, Y_train)     # 计算两者的误差
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_loss.append(loss)
    Y_pred = torch.max(torch.nn.functional.softmax(ypred), 1)[1]

    model.eval()
    with torch.no_grad():
        ypred_test = model(X_test)
        val_loss.append(loss_fn(ypred_test, Y_test))
        Y_pred_test = torch.max(torch.nn.functional.softmax(ypred_test), 1)[1]
        AUC.append(metrics.roc_auc_score(Y_test, torch.nn.functional.softmax(ypred_test)[:,1]))
        Accuracy.append(metrics.accuracy_score(Y_test, Y_pred_test))
        Recall.append(metrics.recall_score(Y_test, Y_pred_test))
        F1_score.append(metrics.f1_score(Y_test, Y_pred_test))
        Precesion.append(metrics.precision_score(Y_test, Y_pred_test))

print(max(AUC))
print(max(Accuracy))
print(max(Recall))
print(max(F1_score))
print(max(Precesion))

#绘制loss曲线
plt.title('loss Function Curve')  # 图片标题
plt.xlabel('epoch')  # x轴变量名称
plt.ylabel('loss')  # y轴变量名称
plt.plot(val_loss, label="$Loss$")  # 逐点画出train_loss_results值并连线，连接图标是loss
plt.legend()  # 画出曲线坐标
plt.show()  # 画出图像

# 绘制Accuracy曲线
plt.title('AUC')  # 图片标题
plt.xlabel('epoch')  # x轴变量名称
plt.ylabel('AUC')  # y轴变量名称
plt.plot(AUC, label="$AUC$")  # 逐点画出test_acc值并连线，连接图标是Accuracy
plt.legend()
plt.show()

torch.save(model.state_dict(), 'DNN_features.pt')

model.eval()
with torch.no_grad():
    data_test = pd.read_csv('G:/cooperate/yuedi/pubchem10000_model_input_dataclean_features.txt', sep = '\t')
    data_test_x = data_test.drop(['GOID', 'CID', 'Pairs'], axis=1)
    data_test_x = torch.tensor(data_test_x.values, dtype=torch.float)
    data_test_pred = model(data_test_x)
    probability =  torch.nn.functional.softmax(data_test_pred)[:,1]
    data_test_Y_pred = torch.max(torch.nn.functional.softmax(data_test_pred), 1)[1]
    data_test['probability'] = probability.detach().numpy()
    data_test['Pred'] = data_test_Y_pred.detach().numpy()
    data_test.to_csv('G:/cooperate/yuedi/pubchem10000_model_input_dataclean_features_pred.xls', index=False, sep='\t')
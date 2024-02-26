import warnings
warnings.filterwarnings("ignore")
import torch
import pandas as pd
import sys
import numpy as np

#train data mean and std
ori_data = pd.read_csv("model255_simi_input_Inf.csv")
selected_features = pd.read_csv('model255_simi_input_dataclean_features.txt', sep = '\t')
ori_data_features = ori_data.loc[:, selected_features.columns[:-1]]
all_median = []
all_mean = []
all_std = []
for i in range(ori_data_features.shape[1]-3):
    all_median.append(np.median(ori_data_features.iloc[:, i].dropna()))
    ori_data_features.iloc[:, i].fillna(value=np.median(ori_data_features.iloc[:, i].dropna()), inplace=True)
    all_mean.append(np.mean(ori_data_features.iloc[:, i]))
    all_std.append(np.std(ori_data_features.iloc[:, i]))
ori_data_features_norm = (ori_data_features.iloc[:, 0:184]-all_mean)/all_std

selected_features.rename(columns = {'MDEO.12':'MDEO-12','SC.5':'SC-5','MDEO.11':'MDEO-11','WTPT.5':'WTPT-5', 'BCUTp.1l':'BCUTp-1l','AVP.5':'AVP-5', 'BCUTc.1l':'BCUTc-1l'}, inplace = True)

#pre-process input data
input_data = pd.read_csv(sys.argv[1], sep='\t')
# input_data = pd.read_csv('prepare_input.txt', sep = '\t')
input_data_features = input_data.loc[:, selected_features.columns[:-1]]

for i in range(input_data_features.shape[1]-3):
    if sum(input_data_features.iloc[:, i].isnull()) > 0:
        input_data_features.iloc[:, i].fillna(value=all_median[i], inplace=True)
input_data_features_norm = (input_data_features.iloc[:, 0:184]-all_mean)/all_std
input_data_features_norm = pd.concat([input_data_features_norm, input_data_features.iloc[:, 184:187]], axis=1)

class DNNModel(torch.nn.Module):
    def __init__(self):
        super(DNNModel, self).__init__()

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

model = DNNModel()
model.load_state_dict(torch.load('DNN_features.pt'))
model.eval()
with torch.no_grad():
    data_test = input_data_features_norm
    data_test_x = data_test.drop(['GOID', 'CID', 'Pairs'], axis=1)
    data_test_x = torch.tensor(data_test_x.values, dtype=torch.float)
    data_test_pred = model(data_test_x)
    probability =  torch.nn.functional.softmax(data_test_pred)[:,1]
    data_test_Y_pred = torch.max(torch.nn.functional.softmax(data_test_pred), 1)[1]
    data_test['probability'] = probability.detach().numpy().round(4)
    data_test['Pred'] = data_test_Y_pred.detach().numpy()
    data_test.to_csv(sys.argv[1] + '_Probility_Pred.xls', index=False, sep='\t')

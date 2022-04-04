import torch
from torch.utils.data.dataset import Dataset
import scipy.io as scio
from sklearn import preprocessing
device = torch.device("cuda:0")
import torch.utils.data as Data

def dataset_read(name):
    if name == 'MSRCv1':
        path = './dataset/MSRC_v1.mat'
        data = scio.loadmat(path)

        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        X11 = min_max_scaler.fit_transform(data['fea'][0][0].astype(float))
        X22 = min_max_scaler.fit_transform(data['fea'][0][1].astype(float))
        X33 = min_max_scaler.fit_transform(data['fea'][0][2].astype(float))
        X44 = min_max_scaler.fit_transform(data['fea'][0][3].astype(float))
        X55 = min_max_scaler.fit_transform(data['fea'][0][4].astype(float))

        X1 = torch.tensor(X11, dtype=torch.float32).to(device)
        X2 = torch.tensor(X22, dtype=torch.float32).to(device)
        X3 = torch.tensor(X33, dtype=torch.float32).to(device)
        X4 = torch.tensor(X44, dtype=torch.float32).to(device)
        X5 = torch.tensor(X55, dtype=torch.float32).to(device)

        Y = data['gt'] - 1
        return X11, X22, X33, X44, X55, X1, X2, X3, X4, X5, Y
    if name == 'Yale':
        path = './dataset/Yale.mat'
        data = scio.loadmat(path)

        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        X11 = min_max_scaler.fit_transform(data['fea'][0][0].astype(float))
        X22 = min_max_scaler.fit_transform(data['fea'][0][1].astype(float))
        X33 = min_max_scaler.fit_transform(data['fea'][0][2].astype(float))

        X1 = torch.tensor(X11, dtype=torch.float32).to(device)
        X2 = torch.tensor(X22, dtype=torch.float32).to(device)
        X3 = torch.tensor(X33, dtype=torch.float32).to(device)

        Y = data['gt'] - 1
        return X11, X22, X33, X1, X2, X3, Y
    if name == 'NUS':
        path = './dataset/NUS_WIDE.mat'
        data = scio.loadmat(path)

        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        X11 = min_max_scaler.fit_transform(data['fea'][0][0].astype(float))
        X22 = min_max_scaler.fit_transform(data['fea'][0][1].astype(float))
        X33 = min_max_scaler.fit_transform(data['fea'][0][2].astype(float))
        X44 = min_max_scaler.fit_transform(data['fea'][0][3].astype(float))
        X55 = min_max_scaler.fit_transform(data['fea'][0][4].astype(float))

        X1 = torch.tensor(X11, dtype=torch.float32).to(device)
        X2 = torch.tensor(X22, dtype=torch.float32).to(device)
        X3 = torch.tensor(X33, dtype=torch.float32).to(device)
        X4 = torch.tensor(X44, dtype=torch.float32).to(device)
        X5 = torch.tensor(X55, dtype=torch.float32).to(device)

        Y = data['gt'] - 1
        return X11, X22, X33, X44, X55, X1, X2, X3, X4, X5, Y
    if name == 'ALOI':
        path = './dataset/ALOI_100.mat'
        data = scio.loadmat(path)

        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        X11 = min_max_scaler.fit_transform(data['fea'][0][0].astype(float))
        X22 = min_max_scaler.fit_transform(data['fea'][0][1].astype(float))
        X33 = min_max_scaler.fit_transform(data['fea'][0][2].astype(float))
        X44 = min_max_scaler.fit_transform(data['fea'][0][3].astype(float))

        X1 = torch.tensor(X11, dtype=torch.float32).to(device)
        X2 = torch.tensor(X22, dtype=torch.float32).to(device)
        X3 = torch.tensor(X33, dtype=torch.float32).to(device)
        X4 = torch.tensor(X44, dtype=torch.float32).to(device)

        Y = data['gt'] - 1
        return X11, X22, X33, X44, X1, X2, X3, X4, Y
    if name == 'Caltech101':
        path = './dataset/Caltech101.mat'
        data = scio.loadmat(path)

        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        # labels = data['gt']-1
        X11 = min_max_scaler.fit_transform(data['fea'][0][0].astype(float))
        X22 = min_max_scaler.fit_transform(data['fea'][0][1].astype(float))
        X33 = min_max_scaler.fit_transform(data['fea'][0][2].astype(float))
        X44 = min_max_scaler.fit_transform(data['fea'][0][3].astype(float))
        X55 = min_max_scaler.fit_transform(data['fea'][0][4].astype(float))
        X66 = min_max_scaler.fit_transform(data['fea'][0][5].astype(float))

        X1 = torch.tensor(X11, dtype=torch.float32).to(device)
        X2 = torch.tensor(X22, dtype=torch.float32).to(device)
        X3 = torch.tensor(X33, dtype=torch.float32).to(device)
        X4 = torch.tensor(X44, dtype=torch.float32).to(device)
        X5 = torch.tensor(X55, dtype=torch.float32).to(device)
        X6 = torch.tensor(X66, dtype=torch.float32).to(device)

        Y = data['gt'] - 1
        return X11, X22, X33, X44, X55, X66, X1, X2, X3, X4, X5, X6, Y


class DL_MSRCv1(Dataset):
    def __init__(self,name):
        self.name = name
        root = './dataset/'
        self.labels, self.feats1, self.feats2, self.feats3, self.feats4, self.feats5 = self.get_data()
        assert len(self.labels) == len(self.feats1) == len(self.feats2) == len(self.feats3) == len(self.feats4) == len(
            self.feats5)
        if len(self.feats1) == 0:
            raise (RuntimeError("Found zero feats in the directory: " + root))

        self.feats1_ = torch.tensor(self.feats1, requires_grad=True).to(device)
        self.labels_ = torch.tensor(self.labels).to(device)
        self.feats2_ = torch.tensor(self.feats2, requires_grad=True).to(device)
        self.feats3_ = torch.tensor(self.feats3, requires_grad=True).to(device)
        self.feats4_ = torch.tensor(self.feats4, requires_grad=True).to(device)
        self.feats5_ = torch.tensor(self.feats5, requires_grad=True).to(device)

    def __getitem__(self, index):
        x1 = self.feats1_[index, :]
        y = self.labels_[index]
        x2 = self.feats2_[index, :]
        x3 = self.feats3_[index, :]
        x4 = self.feats4_[index, :]
        x5 = self.feats5_[index, :]
        return x1, y, x2, x3, x4, x5

    def __len__(self):
        return self.feats1.shape[0]

    def __getlen__(self):
        return [self.feats1.shape[1], self.feats2.shape[1], self.feats3.shape[1], self.feats4.shape[1],
                self.feats5.shape[1]]

    def get_data(self):
        X1, X2, X3, X4, X5, _, _, _, _, _, Y = dataset_read(self.name)
        return Y, X1, X2, X3, X4, X5


class DL_Yale(Dataset):
    def __init__(self,name):
        self.name = name
        root = './dataset/'
        self.labels, self.feats1, self.feats2, self.feats3 = self.get_data()
        assert len(self.labels) == len(self.feats1) == len(self.feats2) == len(self.feats3)
        if len(self.feats1) == 0:
            raise (RuntimeError("Found zero feats in the directory: " + root))

        self.feats1_ = torch.tensor(self.feats1, requires_grad=True).to(device)
        self.labels_ = torch.tensor(self.labels).to(device)
        self.feats2_ = torch.tensor(self.feats2, requires_grad=True).to(device)
        self.feats3_ = torch.tensor(self.feats3, requires_grad=True).to(device)

    def __getitem__(self, index):
        x1 = self.feats1_[index, :]
        y = self.labels_[index]
        x2 = self.feats2_[index, :]
        x3 = self.feats3_[index, :]
        return x1, y, x2, x3

    def __len__(self):
        return self.feats1.shape[0]

    def __getlen__(self):
        return [self.feats1.shape[1], self.feats2.shape[1], self.feats3.shape[1]]

    def get_data(self):

        X1, X2, X3, _, _, _, Y = dataset_read(self.name)
        return Y, X1, X2, X3


class DL_NUS(Dataset):
    def __init__(self,name):
        self.name = name
        root = './dataset/'
        self.labels, self.feats1, self.feats2, self.feats3, self.feats4, self.feats5 = self.get_data()
        assert len(self.labels) == len(self.feats1) == len(self.feats2) == len(self.feats3) == len(self.feats4) == len(
            self.feats5)
        if len(self.feats1) == 0:
            raise (RuntimeError("Found zero feats in the directory: " + root))

        self.feats1_ = torch.tensor(self.feats1, requires_grad=True).to(device)
        self.labels_ = torch.tensor(self.labels).to(device)
        self.feats2_ = torch.tensor(self.feats2, requires_grad=True).to(device)
        self.feats3_ = torch.tensor(self.feats3, requires_grad=True).to(device)
        self.feats4_ = torch.tensor(self.feats4, requires_grad=True).to(device)
        self.feats5_ = torch.tensor(self.feats5, requires_grad=True).to(device)

    def __getitem__(self, index):
        x1 = self.feats1_[index, :]
        y = self.labels_[index]
        x2 = self.feats2_[index, :]
        x3 = self.feats3_[index, :]
        x4 = self.feats4_[index, :]
        x5 = self.feats5_[index, :]
        return x1, y, x2, x3, x4, x5

    def __len__(self):
        return self.feats1.shape[0]

    def __getlen__(self):
        return [self.feats1.shape[1], self.feats2.shape[1], self.feats3.shape[1], self.feats4.shape[1],
                self.feats5.shape[1]]

    def get_data(self):
        X1, X2, X3, X4, X5, _, _, _, _, _, Y = dataset_read(self.name)
        return Y, X1, X2, X3, X4, X5


class DL_ALOI(Dataset):
    def __init__(self,name):
        self.name = name
        self.labels, self.feats1, self.feats2, self.feats3, self.feats4 = self.get_data()

        self.feats1_ = torch.tensor(self.feats1, requires_grad=True).to(device)
        self.labels_ = torch.tensor(self.labels).to(device)
        self.feats2_ = torch.tensor(self.feats2, requires_grad=True).to(device)
        self.feats3_ = torch.tensor(self.feats3, requires_grad=True).to(device)
        self.feats4_ = torch.tensor(self.feats4, requires_grad=True).to(device)

    def __getitem__(self, index):
        x1 = self.feats1_[index, :]
        y = self.labels_[index]
        x2 = self.feats2_[index, :]
        x3 = self.feats3_[index, :]
        x4 = self.feats4_[index, :]
        return x1, y, x2, x3, x4

    def __len__(self):
        return self.feats1.shape[0]

    def __getlen__(self):
        return [self.feats1.shape[1], self.feats2.shape[1], self.feats3.shape[1], self.feats4.shape[1]]

    def get_data(self):
        X1, X2, X3, X4, _, _, _, _, Y = dataset_read(self.name)
        return Y, X1, X2, X3, X4


class DL_Caltech101(Dataset):
    def __init__(self,name):
        self.name = name
        root = './dataset/'
        self.labels, self.feats1, self.feats2, self.feats3, self.feats4, self.feats5, self.feats6 = self.get_data()
        assert len(self.labels) == len(self.feats1) == len(self.feats2) == len(self.feats3) == len(self.feats4) == len(
            self.feats5) == len(self.feats6)
        if len(self.feats1) == 0:
            raise (RuntimeError("Found zero feats in the directory: " + root))

        self.feats1_ = torch.tensor(self.feats1, requires_grad=True).to(device)
        self.labels_ = torch.tensor(self.labels).to(device)
        self.feats2_ = torch.tensor(self.feats2, requires_grad=True).to(device)
        self.feats3_ = torch.tensor(self.feats3, requires_grad=True).to(device)
        self.feats4_ = torch.tensor(self.feats4, requires_grad=True).to(device)
        self.feats5_ = torch.tensor(self.feats5, requires_grad=True).to(device)
        self.feats6_ = torch.tensor(self.feats6, requires_grad=True).to(device)

    def __getitem__(self, index):
        x1 = self.feats1_[index, :]
        y = self.labels_[index]
        x2 = self.feats2_[index, :]
        x3 = self.feats3_[index, :]
        x4 = self.feats4_[index, :]
        x5 = self.feats5_[index, :]
        x6 = self.feats6_[index, :]
        return x1, y, x2, x3, x4, x5, x6

    def __len__(self):
        return self.feats1.shape[0]

    def __getlen__(self):
        return [self.feats1.shape[1], self.feats2.shape[1], self.feats3.shape[1],
                self.feats4.shape[1], self.feats5.shape[1], self.feats6.shape[1]]

    def get_data(self):
        X1, X2, X3, X4, X5, X6, _, _, _, _, _, _, Y = dataset_read(self.name)
        return Y, X1, X2, X3, X4, X5, X6
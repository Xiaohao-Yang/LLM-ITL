import numpy as np
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from scipy import sparse


class CustomDataset(Dataset):

    def __init__(self, dataname='20ng', mode='train'):
        self.mode = mode
        data = sio.loadmat('datasets/%s.mat' % dataname)

        voc = data['voc'].reshape(-1).tolist()
        voc = [v[0] for v in voc]
        self.voc = voc

        if mode == 'train':
            self.data = sparse2dense(data['bow_train'])
            self.label = data['label_train'].reshape(-1,)
        elif mode == 'test':
            self.data = sparse2dense(data['bow_test'])
            self.label = data['label_test'].reshape(-1,)

    def __getitem__(self, index):
        try:
            bow = np.squeeze(self.data[index].toarray())
        except:
            bow = np.squeeze(self.data[index])
        return bow, np.squeeze(self.label[index])

    def __len__(self):
        return self.data.shape[0]


def dataloader(dataname='20ng_6', mode='train', batch_size=500, shuffle=True, drop_last=False, num_workers=4):
    dataset = CustomDataset(dataname=dataname, mode=mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last), dataset.voc

            
def sparse2dense(input_matrix):
    if sparse.isspmatrix(input_matrix):
        input_matrix = input_matrix.toarray()
    input_matrix = input_matrix.astype('float32')
    return input_matrix





import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy import io as sio
from scipy import sparse
import gensim.downloader as api


class TextData:
    def __init__(self, dataset, batch_size):
        # train_data: NxV
        # test_data: Nxv
        # word_emeddings: VxD
        # vocab: V, ordered by word id.
        self.data_name = dataset
        self.train_data, self.test_data, self.train_labels, self.test_labels, self.vocab, self.word_embeddings = self.load_data()
        self.vocab_size = len(self.vocab)

        print("===>train_size: ", self.train_data.shape[0])
        print("===>test_size: ", self.test_data.shape[0])
        print("===>vocab_size: ", self.vocab_size)
        print("===>average length: {:.3f}".format(self.train_data.sum(1).sum() / self.train_data.shape[0]))
        print("===>#label: ", len(np.unique(self.train_labels)))

        self.train_data = torch.from_numpy(self.train_data)
        self.test_data = torch.from_numpy(self.test_data)
        if torch.cuda.is_available():
            self.train_data = self.train_data.cuda()
            self.test_data = self.test_data.cuda()

        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)

    def load_data(self):
        data_dict = sio.loadmat('datasets/%s.mat' % self.data_name)
        train_data = sparse2dense(data_dict['bow_train'])
        test_data = sparse2dense(data_dict['bow_test'])
        voc = data_dict['voc'].reshape(-1).tolist()
        voc = [v[0] for v in voc]

        print('Loading word embeddings ...')
        model_glove = api.load("glove-wiki-gigaword-50")
        print('Loading done!')
        word_embeddings = get_voc_embeddings(voc, model_glove)

        train_labels = data_dict['label_train'].reshape(-1,).astype('int64')
        test_labels = data_dict['label_test'].reshape(-1,).astype('int64')

        return train_data, test_data, train_labels, test_labels, voc, word_embeddings



def sparse2dense(input_matrix):
    if sparse.isspmatrix(input_matrix):
        input_matrix = input_matrix.toarray()
    input_matrix = input_matrix.astype('float32')
    return input_matrix


def get_voc_embeddings(voc, embedding_model):
    word_embeddings = []
    for v in voc:
        word_embeddings.append(embedding_model[v])
    word_embeddings = np.array(word_embeddings).astype('float32')
    return word_embeddings
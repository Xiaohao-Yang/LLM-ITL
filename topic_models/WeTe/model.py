import torch.nn as nn
import torch.nn.functional as F
import pickle
from .Utils import *


class Infer_Net(nn.Module):
    """
    Weibull inference network for topic proportion
    """
    def __init__(self, v=2000, d_hidden=256, k=100):
        super(Infer_Net, self).__init__()
        self.v = v
        self.d_hidden = d_hidden
        self.k = k
        self.encoder = nn.Sequential(
            nn.Linear(self.v, self.d_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.d_hidden, self.d_hidden),
            nn.ReLU(),
            nn.Linear(self.d_hidden, 2 * k),
            nn.Softplus(),
        )

    def reparameterize(self, wei_shape, wei_scale, sample_num=5):
        """
        :param wei_shape: batch, k
        :param wei_scale: batch, k
        :return: Weibull reparameterization
        """
        eps = torch.rand(sample_num, wei_shape.shape[0], wei_shape.shape[1], device=wei_shape.device)
        theta = torch.unsqueeze(wei_scale, axis=0).repeat(sample_num, 1, 1) * torch.pow(-torch.log(eps+1e-10),
                                    torch.unsqueeze(1 / wei_shape, axis=0).repeat(sample_num, 1, 1))
        return torch.mean(torch.clamp(theta, 1e-10, 100.0), dim=0, keepdim=False)   ### for Nan case

    def forward(self, x):
        """
        :param x: document bow vector, batch, v
        :return: unnormalized topic proportions
        """
        wei_shape, wei_scale = torch.chunk(self.encoder(x), 2, dim=-1)
        wei_shape = torch.clamp(wei_shape, 0.1, 100.0)
        wei_scale = torch.clamp(wei_scale, 1e-4, 1e4)
        theta = self.reparameterize(wei_shape, wei_scale)
        return theta


class WeTe(nn.Module):
    """
    WeTe implement in https://arxiv.org/abs/2203.01570
    """
    def __init__(self, args, voc=None):
        super(WeTe, self).__init__()
        self.topic_k = args.n_topic
        self.voc_size = args.vocsize
        self.h = args.embedding_dim
        self.beta = args.beta
        self.epsilon = args.epsilon
        self.real_min = torch.tensor(1e-30)
        self.init_alpha = args.init_alpha
        self.device = args.device
        self.voc = voc

        self.topic_id = torch.tensor([[i] for i in range(self.topic_k)], device=self.device)
        self.word_id = torch.tensor([[i] for i in range(self.voc_size)], device=self.device)
        self.topic_layer = nn.Embedding(self.topic_k, self.h).to(self.device)
        self.word_layer = nn.Embedding(self.voc_size, self.h).to(self.device)
        self.InferNet = Infer_Net(v=self.voc_size, k=self.topic_k)

        self.init_topic(glove=args.glove)
        self.update_embeddings()

    def init_topic(self, glove=None):
        """
        :param glove: Path to pretrained glove embedding
        :return:
        """
        if glove is not None:
            print(f'Load pretrained glove embeddings from : {glove}')
            word_e = np.array(np.random.rand(self.voc_size, self.h) * 0.01, dtype=np.float32)
            num_trained = 0
            for line in open(glove, encoding='UTF-8').readlines():
                sp = line.split()
                if len(sp) == self.h + 1:
                    if sp[0] in self.voc:
                        num_trained += 1
                        word_e[self.voc.index(sp[0])] = [float(x) for x in sp[1:]]
            print(f'num-trained in voc_size: {num_trained}|{self.voc_size}: {1.0 * num_trained / self.voc_size}')
        else:
            print(f'initialize word embedding from N(0, 0.02)')
            word_e = np.random.normal(0, 0.02, size=(self.voc_size, self.h))

        if self.init_alpha:
            cluster_center = cluster_kmeans(word_e, n=self.topic_k)
            self.topic_layer = self.topic_layer.from_pretrained(torch.from_numpy(cluster_center).float(), freeze=False).to(self.device)
        else:
            topic_e = np.random.normal(0, 0.5, size=(self.topic_k, self.h))
            self.topic_layer = self.topic_layer.from_pretrained(torch.from_numpy(topic_e).float(), freeze=False).to(self.device)
        self.word_layer = self.word_layer.from_pretrained(torch.from_numpy(word_e).float(), freeze=True).to(self.device)

    def save_embeddings(self, path='out.pkl'):
        word_e = self.rho.cpu().detach().numpy()
        topic_e = self.alpha.cpu().detach().numpy()
        with open(path, 'wb') as f:
            pickle.dump([word_e, topic_e], f)

    def update_embeddings(self):
        self.rho = self.word_layer(self.word_id).squeeze()
        self.alpha = self.topic_layer(self.topic_id).squeeze()

    def cal_phi(self):
        inner_p = torch.matmul(self.rho, self.alpha.t())
        return F.softmax(inner_p, dim=0)

    def cost_ct(self, inner_p, cost_c, x, theta):
        """
        :param inner_p: v, k
        :param cost_c: v, k
        :param x: batch of sequential words
        :param theta: batch, k, topic proportions
        :return: bi-direction cost
        """
        dis_d = torch.clamp(torch.exp(inner_p), 1e-30, 1e10)
        forward_cost = 0.
        backward_cost = 0.
        theta_norm = F.softmax(theta, dim=-1)
        for each, each_theta in zip(x, theta_norm):
            forward_doc_dis = dis_d[each] * each_theta[None, :]   ## N_j * K
            doc_dis = dis_d[each]  ## N_J * K
            forward_pi = forward_doc_dis / (torch.sum(forward_doc_dis, dim=1, keepdim=True) + self.real_min)  ### N_j, K
            backward_pi = doc_dis / (torch.sum(doc_dis, dim=0, keepdim=True) + self.real_min)  ### N_j, K
            forward_cost += (cost_c[each] * forward_pi).sum(1).mean()
            backward_cost += ((cost_c[each] * backward_pi).sum(0) * each_theta).sum()
        return forward_cost, backward_cost

    def Poisson_likelihood(self, x, re_x):
        """
        :param x: batch of bow vector
        :param re_x: \Phi \times \theta
        :return: Negative log of poisson likelihoood
        """
        return -(x * torch.log(re_x + 1e-10) - re_x - torch.lgamma(x + 1.0)).sum(-1).mean()

    def forward(self, x, bow):
        theta = self.InferNet(bow)
        self.update_embeddings()
        phi = self.cal_phi()
        ## calculate distance between word and topic embeddings
        inner_p = torch.matmul(self.rho, self.alpha.t())
        cost_c = torch.clamp(torch.exp(-inner_p), 1e-30, 1e10)
        forward_cost, backward_cost = self.cost_ct(inner_p, cost_c, x, theta)
        re_x = torch.matmul(phi, theta.t())
        TM_cost = self.Poisson_likelihood(bow, re_x.t())
        loss = self.beta * forward_cost + (1-self.beta) * backward_cost + self.epsilon * TM_cost
        return loss, forward_cost, backward_cost, TM_cost, theta
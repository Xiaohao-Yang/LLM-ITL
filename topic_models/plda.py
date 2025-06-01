import sys
sys.path.append("../LLM-ITL")
import os
import argparse
import numpy as np
import multiprocessing as mp
import datetime

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from torch.nn import functional as F

import gensim.downloader as api
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate import generate_one_pass
from topic_models.refine_funcs import compute_refine_loss, save_llm_topics
from collections import OrderedDict
import scipy.io as sio
from utils import sparse2dense
from tqdm import tqdm
from topic_models.hyperparameters import hyperparamters


class BOWDataset(Dataset):
    def __init__(self, X, idx2token):
        self.X = X
        self.idx2token = idx2token

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        X = torch.FloatTensor(self.X[i])
        return {'X': X}


class InferenceNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes,
                 activation='softplus', dropout=0.2):
        super(InferenceNetwork, self).__init__()
        assert isinstance(input_size, int), "input_size must by type int."
        assert isinstance(output_size, int), "output_size must be type int."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu'], \
            "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout

        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()

        self.input_layer = nn.Linear(input_size, hidden_sizes[0])

        self.hiddens = nn.Sequential(OrderedDict([
            ('l_{}'.format(i), nn.Sequential(nn.Linear(h_in, h_out), self.activation))
            for i, (h_in, h_out) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]))]))

        self.f_mu = nn.Linear(hidden_sizes[-1], output_size)
        self.f_mu_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.f_sigma = nn.Linear(hidden_sizes[-1], output_size)
        self.f_sigma_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.dropout_enc = nn.Dropout(p=self.dropout)

    def forward(self, x):
        """Forward pass."""
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hiddens(x)
        x = self.dropout_enc(x)
        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))

        return mu, log_sigma


class DecoderNetwork(nn.Module):
    def __init__(self, input_size, n_components=10, model_type='prodLDA',
                 hidden_sizes=(100,100), activation='softplus', dropout=0.2,
                 learn_priors=True):
        super(DecoderNetwork, self).__init__()
        assert isinstance(input_size, int), "input_size must by type int."
        assert isinstance(n_components, int) and n_components > 0, \
            "n_components must be type int > 0."
        assert model_type in ['prodLDA', 'LDA'], \
            "model type must be 'prodLDA' or 'LDA'"
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu'], \
            "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."

        self.input_size = input_size
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors

        self.inf_net = InferenceNetwork(
            input_size, n_components, hidden_sizes, activation)

        topic_prior_mean = 0.0
        self.prior_mean = torch.tensor(
            [topic_prior_mean] * n_components)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)

        topic_prior_variance = 1. - (1. / self.n_components)
        self.prior_variance = torch.tensor(
            [topic_prior_variance] * n_components)
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()
        if self.learn_priors:
            self.prior_variance = nn.Parameter(self.prior_variance)

        self.beta = torch.Tensor(n_components, input_size)
        if torch.cuda.is_available():
            self.beta = self.beta.cuda()
        self.beta = nn.Parameter(self.beta)
        nn.init.xavier_uniform_(self.beta)

        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)

        # dropout on theta
        self.drop_theta = nn.Dropout(p=self.dropout)

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        """Forward pass."""
        # batch_size x n_components
        posterior_mu, posterior_log_sigma = self.inf_net(x)
        posterior_sigma = torch.exp(posterior_log_sigma)

        # generate samples from theta
        theta = F.softmax(
            self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)
        theta = self.drop_theta(theta)

        # prodLDA vs LDA
        if self.model_type == 'prodLDA':
            # in: batch_size x input_size x n_components
            word_dist = F.softmax(
                self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1)
            # word_dist: batch_size x input_size
        elif self.model_type == 'LDA':
            # simplex constrain on Beta
            beta = F.softmax(self.beta_batchnorm(self.beta), dim=1)
            word_dist = torch.matmul(theta, beta)
            # word_dist: batch_size x input_size

        return self.prior_mean, self.prior_variance, \
            posterior_mu, posterior_sigma, posterior_log_sigma, word_dist, theta


class AVITM(object):
    def __init__(self, input_size, n_components=10, model_type='prodLDA',
                 hidden_sizes=(100, 100), activation='softplus', dropout=0.2,
                 learn_priors=True, batch_size=64, lr=2e-3, momentum=0.99,
                 solver='adam', num_epochs=100, reduce_on_plateau=False):
        self.input_size = input_size
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.solver = solver
        self.num_epochs = num_epochs
        self.reduce_on_plateau = reduce_on_plateau

        # init inference avitm network
        self.model = DecoderNetwork(input_size, n_components, model_type, hidden_sizes, activation,
                                    dropout, learn_priors)
        # init optimizer
        if self.solver == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, betas=(self.momentum, 0.99))
        elif self.solver == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=self.momentum)
        # init lr scheduler
        if self.reduce_on_plateau:
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=10)

        # performance attributes
        self.best_loss_train = float('inf')
        # training atributes
        self.model_dir = None
        self.train_data = None
        self.nn_epoch = None
        # learned topics
        self.best_components = None
        # Use cuda if available
        if torch.cuda.is_available():
            self.USE_CUDA = True
        else:
            self.USE_CUDA = False
        if self.USE_CUDA:
            self.model = self.model.cuda()


    def _loss(self, inputs, word_dists, prior_mean, prior_variance,
              posterior_mean, posterior_variance, posterior_log_variance):
        # KL term
        # var division term
        var_division = torch.sum(posterior_variance / prior_variance, dim=1)
        # diff means term
        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum(
            (diff_means * diff_means) / prior_variance, dim=1)
        # logvar det division term
        logvar_det_division = \
            prior_variance.log().sum() - posterior_log_variance.sum(dim=1)
        # combine terms
        KL = 0.5 * (
                var_division + diff_term - self.n_components + logvar_det_division)
        # Reconstruction term
        RL = -torch.sum(inputs * torch.log(word_dists + 1e-10), dim=1)
        return KL, RL


    def fit(self, datas, args):
        if args.llm_itl:
            print('Loading embedding model and LLM ...')
            embedding_model = api.load("glove-wiki-gigaword-50")
            llm = AutoModelForCausalLM.from_pretrained(args.llm, trust_remote_code=True, torch_dtype=torch.float16).cuda()
            tokenizer = AutoTokenizer.from_pretrained(args.llm, padding_side='left')
            tokenizer.pad_token = tokenizer.eos_token
            print('Loading done!')

        self.train_data = datas['train_data']
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,num_workers=mp.cpu_count())

        # to save model checkpoint
        run_name = '%s_%s_K%s_seed%s_useLLM-%s' % (args.name, args.dataset, args.n_topic, args.seed, args.llm_itl)

        checkpoint_folder = "save_models/%s" % run_name
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)

        # train loop
        for epoch in range(self.num_epochs):
            self.nn_epoch = epoch
            self.model.train()

            s = datetime.datetime.now()
            running_loss = 0.0
            running_kld = 0.0
            running_rec = 0.0
            running_refine = 0.0

            for batch_samples in tqdm(train_loader, desc='Epoch: %s' % (epoch + 1)):
                self.model.zero_grad()
                X = batch_samples['X']
                if self.USE_CUDA:
                    X = X.cuda()

                # forward
                prior_mean, prior_variance, \
                    posterior_mean, posterior_variance, posterior_log_variance, \
                    word_dists, _ = self.model(X)

                # ntm loss
                kld, recons_loss = self._loss(
                    X, word_dists, prior_mean, prior_variance,
                    posterior_mean, posterior_variance, posterior_log_variance)

                # refine loss
                refine_loss = 0
                if args.llm_itl and epoch >= args.warmStep:
                    # get topics
                    beta = self.model.beta
                    top_values, top_idxs = torch.topk(beta, k=args.n_topic_words, dim=1)
                    topic_probas = torch.div(top_values, top_values.sum(dim=-1).unsqueeze(-1))
                    topic_words = []
                    for i in range(top_idxs.shape[0]):
                        topic_words.append([datas['voc'][j] for j in top_idxs[i, :].tolist()])

                    # llm top words and prob
                    suggest_topics, suggest_words = generate_one_pass(llm, tokenizer, topic_words, datas['token2idx'],
                                                                          instruction_type=args.instruction,
                                                                          batch_size=args.inference_bs,
                                                                          max_new_tokens=args.max_new_tokens)

                    # compute refine loss
                    refine_loss = compute_refine_loss(topic_probas, topic_words, suggest_topics, suggest_words, embedding_model)

                # compute loss
                loss = kld.mean() + recons_loss.mean()
                if refine_loss > 0:
                    loss += refine_loss * args.refine_weight

                loss.backward()
                self.optimizer.step()

                # update records
                running_loss += loss.item()
                running_kld += kld.mean().item()
                running_rec += recons_loss.mean().item()
                if refine_loss > 0:
                    running_refine += refine_loss.item()

            e = datetime.datetime.now()
            avg_loss = running_loss / len(train_loader)
            avg_kld = running_kld / len(train_loader)
            avg_rec = running_rec / len(train_loader)
            avg_refine = running_refine / len(train_loader)

            print('| Time : {} |'.format(e - s),
                  '| Epoch train: {:d} |'.format(epoch + 1),
                  '| Total Loss: {:.5f}'.format(avg_loss),
                  '| Rec Loss: {:.5f}'.format(avg_rec),
                  '| KLD Loss: {:.5f}'.format(avg_kld),
                  '| refine Loss: {:.5}'.format(avg_refine))

            if (epoch + 1) % args.eval_step == 0:
                self.model.eval()
                topic_dir = 'save_topics/%s' % run_name
                if not os.path.exists(topic_dir):
                    os.makedirs(topic_dir)

                # save tm topics
                beta = self.model.beta
                _, top_idxs = torch.topk(beta, k=args.n_topic_words, dim=1)
                tm_topics = []
                for i in range(top_idxs.shape[0]):
                    tm_topics.append([datas['voc'][j] for j in top_idxs[i, :].tolist()])

                with open(os.path.join(topic_dir, 'epoch%s_tm_words.txt' % (epoch + 1)), 'w') as file:
                    for item in tm_topics:
                        file.write(' '.join(item) + '\n')

                # save llm topics
                if args.llm_itl and epoch >= args.warmStep:
                    llm_topics_dicts, llm_words_dicts = generate_one_pass(llm, tokenizer, tm_topics, datas['token2idx'],
                                                                              instruction_type=args.instruction,
                                                                              batch_size=args.inference_bs,
                                                                              max_new_tokens=args.max_new_tokens)


                    save_llm_topics(llm_topics_dicts, llm_words_dicts, epoch, topic_dir)

                # save model
                torch.save(self.model, os.path.join(checkpoint_folder, 'epoch-%s.pth' % (epoch + 1)))
                self.model.train()


def train():
    # load data
    data_dict = sio.loadmat('datasets/%s.mat' % args.dataset)
    train_data = sparse2dense(data_dict['bow_train'])
    test_data = sparse2dense(data_dict['bow_test'])
    voc = data_dict['voc'].reshape(-1).tolist()
    voc = [v[0] for v in voc]
    all_data = np.vstack((train_data, test_data))  # to evaluate TC

    # create dataset
    idx2token = {index:word for index, word in enumerate(voc)}
    token2idx = {word:index for index, word in enumerate(voc)}
    train_dataset = BOWDataset(train_data, idx2token)

    # create model
    avitm = AVITM(input_size=len(voc), n_components=args.n_topic, model_type='prodLDA',
                  hidden_sizes=(200,), activation='softplus', dropout=0.2,
                  learn_priors=True, batch_size=args.batch_size, lr=args.lr, momentum=0.9,
                  solver='adam', num_epochs=args.epochs, reduce_on_plateau=False)
    # start training
    datas = {'train_data': train_dataset,
             'voc': voc,
             'all_data': all_data,
             'token2idx': token2idx}

    avitm.fit(datas, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ProdLDA')
    parser.add_argument("--name", type=str, default="PLDA")
    parser.add_argument('--dataset', type=str, default='20News')
    parser.add_argument('--n_topic', type=int, default=50)
    parser.add_argument("--eval_step", default=10, type=int)
    parser.add_argument("--seed", default=1, type=int)

    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--warmStep', default=150, type=int)
    parser.add_argument('--llm_itl', action='store_true')
    parser.add_argument('--llm_step', type=int, default=50)  # the number of epochs for llm refine
    parser.add_argument('--llm', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--refine_weight', type=float, default=200)
    parser.add_argument('--instruction', type=str, default='refine_labelTokenProbs',
                        choices=['refine_labelTokenProbs', 'refine_wordIntrusion'])
    parser.add_argument('--inference_bs', type=int, default=100)
    parser.add_argument('--max_new_tokens', type=int, default=300)

    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--n_topic_words', default=10, type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load hyper-parameters for topic model
    hps = hyperparamters[args.name + '_' + args.dataset]
    args.epochs = hps[0]
    args.lr = hps[1]
    args.batch_size = hps[2]

    args.warmStep = args.epochs - args.llm_step  # Leave X epochs for LLM refinement

    train()
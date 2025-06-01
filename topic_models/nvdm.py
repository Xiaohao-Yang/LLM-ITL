import sys
sys.path.append("../LLM-ITL")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os.path
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from generate import generate_one_pass
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import sparse2dense
from topic_models.refine_funcs import compute_refine_loss, save_llm_topics
import gensim.downloader as api
import random
import scipy.io as sio
from topic_models.hyperparameters import hyperparamters


class FeatDataset(Dataset):
    def __init__(self, data_np, ):
        self.data = data_np
        self.word_count = data_np.sum(1)

    def __getitem__(self, item):
        return self.data[item], self.word_count[item]

    def __len__(self):
        return len(self.data)


class NVDM(nn.Module):
    def __init__(self, vocab_size, n_hidden, n_topic, n_sample):
        super(NVDM, self).__init__()

        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_topic = n_topic
        self.n_sample = n_sample

        # encoder architecture
        # encode doc to vectors
        self.enc_vec = nn.Linear(self.vocab_size, self.n_hidden)
        # get mean of Gaussian distribution
        self.mean = nn.Linear(self.n_hidden, self.n_topic)
        # get log_sigma of Gaussian distribution
        self.log_sigma = nn.Linear(self.n_hidden, self.n_topic)

        # decoder architecture
        self.dec_vec = nn.Linear(self.n_topic, self.vocab_size)

    def encoder(self, x):
        # encode doc to vectors
        enc_vec = torch.tanh(self.enc_vec(x))
        # getting variational parameters
        mean = self.mean(enc_vec)
        log_sigma = self.log_sigma(enc_vec)
        # computing kld
        kld = -0.5 * torch.sum(1 - torch.square(mean) + 2 * log_sigma - torch.exp(2 * log_sigma), 1)
        return mean, log_sigma, kld

    def decoder(self, mean, log_sigma, x):
        # reconstruct doc from encoded vector
        if self.n_sample == 1:  # single sample
            eps = torch.rand(self.batch_size, self.n_topic).cuda()
            doc_vec = torch.mul(torch.exp(log_sigma), eps) + mean
            logits = F.log_softmax(self.dec_vec(doc_vec), dim=1)
            recons_loss = -torch.sum(torch.mul(logits, x), 1)
        # multiple samples
        else:
            eps = torch.rand(self.n_sample * self.batch_size, self.n_topic)
            eps_list = list(eps.view(self.n_sample, self.batch_size, self.n_topic))
            recons_loss_list = []
            for i in range(self.n_sample):
                curr_eps = eps_list[i]
                doc_vec = torch.mul(torch.exp(log_sigma), curr_eps) + mean
                logits = F.log_softmax(self.dec_vec(doc_vec))
                recons_loss_list.append(-torch.sum(torch.mul(logits, x), 1))
            recons_loss_list = torch.tensor(recons_loss_list)
            recons_loss = torch.sum(recons_loss_list, dim=1) / self.n_sample

        return recons_loss, doc_vec

    def forward(self, x):
        self.batch_size = len(x)
        mean, log_sigma, kld = self.encoder(x)
        recons_loss, doc_vec = self.decoder(mean, log_sigma, x)

        return kld, recons_loss, mean, doc_vec

    # get topic-word weights
    def get_beta(self):
        emb = self.dec_vec.weight.T
        return emb


def train():
    # load data
    data_dict = sio.loadmat('datasets/%s.mat' % args.dataset)
    train_data = sparse2dense(data_dict['bow_train'])
    train_dataset = FeatDataset(train_data)

    voc = data_dict['voc'].reshape(-1).tolist()
    voc = [v[0] for v in voc]
    token2idx = {word: index for index, word in enumerate(voc)}

    # load pre-trained models
    if args.llm_itl:
        print('Loading embedding model and LLM ...')
        embedding_model = api.load("glove-wiki-gigaword-50")

        llm = AutoModelForCausalLM.from_pretrained(args.llm, trust_remote_code=True, torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.llm, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        print('Loading done!')

    # create topic model
    model = NVDM(len(voc), args.n_hidden, args.n_topic, args.n_sample).to(device)
    # create data loader
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # create optimiser
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, 0.999))

    # to save model checkpoint
    run_name = ('%s_%s_K%s_seed%s_useLLM-%s' %
                (args.name, args.dataset, args.n_topic, args.seed, args.llm_itl))

    checkpoint_folder = "save_models/%s" % run_name
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    # training loop
    for epoch in range(args.epochs):
        s = datetime.datetime.now()
        running_loss = 0.0
        running_kld = 0.0
        running_rec = 0.0
        running_refine = 0.0

        for data_batch, _ in tqdm(dataloader, desc='Epoch: %s' % (epoch+1)):
            optim.zero_grad()
            data_batch = data_batch.float().cuda()

            # forward
            kld, recons_loss, _, _ = model(data_batch)

            # for llm refine
            refine_loss = 0
            if args.llm_itl and epoch >= args.warmStep:
                # get topics
                beta = model.dec_vec.weight.T
                top_values, top_idxs = torch.topk(beta, k=args.n_topic_words, dim=1)
                topic_probas = torch.div(top_values, top_values.sum(dim=-1).unsqueeze(-1))
                topic_words = []
                for i in range(top_idxs.shape[0]):
                    topic_words.append([voc[j] for j in top_idxs[i,:].tolist()])

                # llm top words and prob
                suggest_topics, suggest_words = generate_one_pass(llm, tokenizer, topic_words, token2idx,
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
            optim.step()

            # update records
            running_loss += loss.item()
            running_kld += kld.mean().item()
            running_rec += recons_loss.mean().item()
            if refine_loss > 0:
                running_refine += refine_loss.item()

        e = datetime.datetime.now()
        avg_loss = running_loss / len(dataloader)
        avg_kld = running_kld/len(dataloader)
        avg_rec = running_rec/len(dataloader)
        avg_refine = running_refine/len(dataloader)

        print('| Time : {} |'.format(e - s),
              '| Epoch train: {:d} |'.format(epoch + 1),
              '| Total Loss: {:.5f}'.format(avg_loss),
              '| Rec Loss: {:.5f}'.format(avg_rec),
              '| KLD Loss: {:.5f}'.format(avg_kld),
              '| refine Loss: {:.5}'.format(avg_refine))


        # evaluation phase
        if (epoch + 1) % args.eval_step == 0:
            model.eval()

            topic_dir = 'save_topics/%s' % run_name
            if not os.path.exists(topic_dir):
                os.makedirs(topic_dir)

            # save tm topics
            beta = model.dec_vec.weight.T
            _, top_idxs = torch.topk(beta, k=args.n_topic_words, dim=1)
            tm_topics = []
            for i in range(top_idxs.shape[0]):
                tm_topics.append([voc[j] for j in top_idxs[i, :].tolist()])

            with open(os.path.join(topic_dir, 'epoch%s_tm_words.txt' % (epoch + 1)), 'w') as file:
                for item in tm_topics:
                    file.write(' '.join(item) + '\n')

            # save llm topics
            if args.llm_itl and epoch >= args.warmStep:
                llm_topics_dicts, llm_words_dicts = generate_one_pass(llm, tokenizer, tm_topics, token2idx,
                                                                          instruction_type=args.instruction,
                                                                          batch_size=args.inference_bs,
                                                                          max_new_tokens=args.max_new_tokens)

                save_llm_topics(llm_topics_dicts, llm_words_dicts, epoch, topic_dir)

            # save model
            torch.save(model, os.path.join(checkpoint_folder, 'epoch-%s.pth' % (epoch + 1)))
            model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NVDM')
    parser.add_argument("--name", type=str, default="NVDM")
    parser.add_argument('--dataset', type=str, default='20News')
    parser.add_argument('--n_topic', type=int, default=50)
    parser.add_argument('--eval_step', default=10, type=int)
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--warmStep', default=150, type=int)
    parser.add_argument('--llm_step', type=int, default=50)  # the number of epochs for llm refine
    parser.add_argument('--llm_itl', action='store_true')
    parser.add_argument('--llm', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--refine_weight', type=float, default=200)
    parser.add_argument('--instruction', type=str, default='refine_labelTokenProbs',
                        choices=['refine_labelTokenProbs', 'refine_wordIntrusion'])
    parser.add_argument('--inference_bs', type=int, default=100)
    parser.add_argument('--max_new_tokens', type=int, default=300)

    parser.add_argument('--n_hidden', default=100, type=int)
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--n_topic_words', default=10, type=int)
    parser.add_argument('--n_sample', default=1, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load hyper-parameters for topic model
    hps = hyperparamters[args.name + '_' + args.dataset]
    args.epochs = hps[0]
    args.lr = hps[1]
    args.batch_size = hps[2]

    args.warmStep = args.epochs - args.llm_step  # Leave X epochs for LLM refinement

    train()
import sys
sys.path.append("../LLM-ITL")
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
import argparse
import torch
import math
import numpy as np
from utils import sparse2dense
from topic_models.refine_funcs import compute_refine_loss, save_llm_topics
import gensim.downloader as api
from scipy import sparse
import os
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate import generate_one_pass
import scipy.io as sio
from topic_models.hyperparameters import hyperparamters


def get_voc_embeddings(voc, embedding_model):
    word_embeddings = []
    for v in voc:
        word_embeddings.append(embedding_model[v])
    word_embeddings = np.array(word_embeddings)
    return word_embeddings

def batch_indices(batch_nb, data_length, batch_size):
    # Batch start and end index
    start = int(batch_nb * batch_size)
    end = int((batch_nb + 1) * batch_size)

    # When there are not enough inputs left, we reuse some to complete the batch
    if end > data_length:
        shift = end - data_length
        start -= shift
        end -= shift

    return start, end


def sinkhorn_torch(M, a, b, lambda_sh, numItermax=5000, stopThr=.5e-2, cuda=False):

    if cuda:
        u = (torch.ones_like(a) / a.size()[0]).double().cuda()
        v = (torch.ones_like(b)).double().cuda()
    else:
        u = (torch.ones_like(a) / a.size()[0])
        v = (torch.ones_like(b))
    if torch.any(torch.isnan(M)):
        print("M M M M M M is nan now")
    K = torch.exp(-M * lambda_sh)
    err = 1
    cpt = 0
    while err > stopThr and cpt < numItermax:
        u = torch.div(a, torch.matmul(K, torch.div(b, torch.matmul(u.t(), K).t())))
        cpt += 1
        if cpt % 20 == 1:
            v = torch.div(b, torch.matmul(K.t(), u))
            u = torch.div(a, torch.matmul(K, v))
            bb = torch.mul(v, torch.matmul(K.t(), u))
            err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

    sinkhorn_divergences = torch.sum(torch.mul(u, torch.matmul(torch.mul(K, M), v)), dim=0)
    return sinkhorn_divergences


class encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class):
        self.K = num_class
        super(encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(hidden_dim, num_class),
            nn.BatchNorm1d(num_class,eps=0.001, momentum=0.001, affine=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)


class myLoss(nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()

    def forward(self, x, doc_topic, doc_word, M, topic_embedding, sh_alpha, rec_loss_weight):
        sh_loss = sinkhorn_torch(M, doc_topic.t(), doc_word.t(), lambda_sh=sh_alpha).mean()

        rec_log_probs = F.log_softmax(torch.matmul(doc_topic, (1-M)), dim=1)

        rec_loss = -torch.mean(torch.sum((rec_log_probs * x), dim=1))

        joint_loss = rec_loss_weight * rec_loss + sh_loss

        return rec_loss, sh_loss, joint_loss


def run_ntsm():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    torch.autograd.set_detect_anomaly(True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # load data
    data_dict = sio.loadmat('datasets/%s.mat' % args.dataset)
    train_data = sparse2dense(data_dict['bow_train'])
    test_data = sparse2dense(data_dict['bow_test'])
    voc = data_dict['voc'].reshape(-1).tolist()
    voc = [v[0] for v in voc]
    token2idx = {word: index for index, word in enumerate(voc)}

    if args.llm_itl:
        # load LLM
        llm = AutoModelForCausalLM.from_pretrained(args.llm,
                                                   trust_remote_code=True,
                                                   torch_dtype=torch.float16
                                                   ).cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.llm, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        print('Loading done!')


    print('Loading glove model ...')
    model_glove = api.load("glove-wiki-gigaword-50")
    print('Loading done!')
    word_embeddings = get_voc_embeddings(voc, model_glove)

    V = data_dict['bow_train'].shape[1]       # voc size
    N = data_dict['bow_train'].shape[0]       # train size
    L = word_embeddings.shape[1]               # embedding dim

    # word embedding
    word_embedding = torch.tensor(word_embeddings, dtype=torch.float32, device=device, requires_grad=True)

    # topic embedding
    topic_embedding = torch.zeros(size=(args.n_topic, L),dtype=torch.float32, requires_grad=True, device=device)
    torch.nn.init.trunc_normal_(topic_embedding, std=0.1)

    # set model
    model = encoder(V, args.hidden_dim, args.n_topic).to(device)
    loss_function = myLoss()
    # optimizer
    optimize_params = list(model.parameters())
    optimize_params.append(topic_embedding)
    optimizer = torch.optim.Adam(optimize_params, lr=args.lr, betas=(0.99, 0.999))

    # to save model checkpoint
    run_name = '%s_%s_K%s_seed%s_useLLM-%s' % (args.name, args.dataset, args.n_topic, args.seed, args.llm_itl)

    checkpoint_folder = "save_models/%s" % run_name
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    # train loop
    nb_batches = int(math.ceil(float(N) / args.batch_size))
    assert nb_batches * args.batch_size >= N
    for epoch in range(args.epochs):
        idxlist = np.random.permutation(N)  # can be used to index train_text
        rec_loss_avg, sh_loss_avg, joint_loss_avg, refine_loss_avg = 0., 0., 0., 0.

        for batch in tqdm(range(nb_batches)):
            optimizer.zero_grad()
            start, end = batch_indices(batch, N, args.batch_size)
            X = train_data[idxlist[start:end]]

            if sparse.isspmatrix(X):
                X = X.toarray()
            X = torch.tensor(X, device=device)

            doc_word = F.softmax(X, dim=1)
            doc_topic = model(X)

            word_embedding_norm = F.normalize(word_embedding, p=2, dim=1)
            topic_embedding_norm = F.normalize(topic_embedding, p=2, dim=1)
            topic_word = torch.matmul(topic_embedding_norm, word_embedding_norm.t())

            M = 1 - topic_word

            rec_loss_batch, sh_rec_loss_batch, joint_loss_batch = \
                loss_function(X, doc_topic, doc_word, M, topic_embedding, args.sh_alpha, args.rec_loss_weight)

            # for llm refine
            refine_loss = 0
            if args.llm_itl and epoch >= args.warmStep:
                # get topics
                beta = topic_word
                top_values, top_idxs = torch.topk(beta, k=args.n_topic_words, dim=1)
                topic_probas = torch.div(top_values, top_values.sum(dim=-1).unsqueeze(-1))
                topic_words = []
                for i in range(top_idxs.shape[0]):
                    topic_words.append([voc[j] for j in top_idxs[i, :].tolist()])

                # llm top words and prob
                suggest_topics, suggest_words = generate_one_pass(llm, tokenizer, topic_words, token2idx,
                                                                      instruction_type=args.instruction,
                                                                      batch_size=args.inference_bs,
                                                                      max_new_tokens=args.max_new_tokens)

                # compute refine loss
                refine_loss = compute_refine_loss(topic_probas, topic_words, suggest_topics, suggest_words, model_glove)

            if refine_loss > 0:
                joint_loss_batch += refine_loss * args.refine_weight

            joint_loss_batch.backward()
            optimizer.step()

            rec_loss_avg += rec_loss_batch.item()
            sh_loss_avg += sh_rec_loss_batch.item()
            joint_loss_avg += joint_loss_batch.item()
            if refine_loss > 0:
                refine_loss_avg += refine_loss.item()


        print('| Epoch train: {:d} |'.format(epoch + 1),
              '| Rec Loss: {:.5f}'.format(rec_loss_avg),
              '| Sinkhorn Loss: {:.5f}'.format(sh_loss_avg),
              '| NSTM Loss: {:.5f}'.format(joint_loss_avg),
              '| Refine Loss: {:.5f}'.format(refine_loss_avg))


        if (epoch + 1) % args.eval_step == 0:
            model.eval()
            topic_dir = 'save_topics/%s' % run_name
            if not os.path.exists(topic_dir):
                os.makedirs(topic_dir)

            # save tm topics
            beta = topic_word.clone().detach().cpu().numpy()
            top_idxs = np.argsort(beta, axis=1)[:, -args.n_topic_words:]
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
    parser = argparse.ArgumentParser(description='NSTM')
    parser.add_argument("--name", type=str, default="NSTM")
    parser.add_argument("--dataset", type=str, default='20News')
    parser.add_argument("--n_topic", default=50, type=int)
    parser.add_argument("--eval_step", default=10, type=int)
    parser.add_argument("--seed", default=1, type=int)

    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument('--warmStep', default=100, type=int)
    parser.add_argument('--llm_itl', action='store_true')
    parser.add_argument('--llm_step', type=int, default=50)  # the number of epochs for llm refine
    parser.add_argument('--llm', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--refine_weight', type=float, default=200)
    parser.add_argument('--instruction', type=str, default='refine_labelTokenProbs',
                        choices=['refine_labelTokenProbs', 'refine_wordIntrusion'])
    parser.add_argument('--inference_bs', type=int, default=100)
    parser.add_argument('--max_new_tokens', type=int, default=300)

    parser.add_argument('--n_topic_words', default=10, type=int)
    parser.add_argument("--rec_loss_weight", default=0.7, type=float)  # 0.07
    parser.add_argument("--hidden_dim", help="Hidden dimension", default=200, type=int)
    parser.add_argument("--batch_size", default=200, type=int)
    parser.add_argument("--sh_iterations", default=50, type=int)
    parser.add_argument("--sh_epsilon", default=0.001, type=float)
    parser.add_argument("--sh_alpha", default=20, type=int)
    args = parser.parse_args()

    # load hyper-parameters for topic model
    hps = hyperparamters[args.name + '_' + args.dataset]
    args.epochs = hps[0]
    args.lr = hps[1]
    args.batch_size = hps[2]

    args.warmStep = args.epochs - args.llm_step  # Leave X epochs for LLM refinement

    run_ntsm()


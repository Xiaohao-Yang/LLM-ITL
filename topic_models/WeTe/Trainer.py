import sys
sys.path.append("../LLM-ITL")
import torch.nn as nn
from tqdm import tqdm
from .Utils import *
import random
import gensim.downloader as api
from transformers import AutoModelForCausalLM, AutoTokenizer
from topic_models.refine_funcs import compute_refine_loss, save_llm_topics
from generate import generate_one_pass


class Trainer(object):
    """
    Trainer for WeTe
    """
    def __init__(self, args, model, voc=None):
        super(Trainer, self).__init__()
        self.model = model.to(args.device)
        self.epoch = args.epochs
        self.data_name = args.dataset
        self.device = args.device
        self.topic_k = args.n_topic
        self.test_every = args.eval_step
        self.train_num = -1
        self.voc = voc
        self.token2idx = {word: index for index, word in enumerate(self.voc)}
        self.args = args
        self.run_name = ('%s_%s_K%s_seed%s_useLLM-%s' %
                         (args.name, args.dataset, args.n_topic, args.seed, args.llm_itl))

        # log_str = 'runs/{}/k_{}'.format(args.dataset, self.topic_k)
        # now = int(round(time.time() * 1000))
        # now_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(now / 1000))
        # self.log_str = log_str + '/' + now_time
        # if not os.path.exists(self.log_str):
        #     os.makedirs(self.log_str)

        self.trainable_params = []
        print('WeTe learnable params:')
        for name, params in self.model.named_parameters():
            if params.requires_grad:
                print(name)
                self.trainable_params.append(params)
        self.optimizer = torch.optim.Adam(self.trainable_params, lr=args.lr, weight_decay=1e-3)

    def train(self, train_loader, test_loader):
        if self.args.llm_itl:
            # Load embedding model
            print('Loading embedding model and LLM ...')
            embedding_model = api.load("glove-wiki-gigaword-50")

            # load LLM
            llm = AutoModelForCausalLM.from_pretrained(self.args.llm,
                                                       trust_remote_code=True,
                                                       torch_dtype=torch.float16
                                                       ).cuda()
            tokenizer = AutoTokenizer.from_pretrained(self.args.llm, padding_side='left')
            tokenizer.pad_token = tokenizer.eos_token
            print('Loading done!')


        for epoch in range(self.epoch):
            tr_loss = []
            tr_forward_cost = []
            tr_backward_cost = []
            tr_tm = []
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))
            self.model.train()


            for j, (bow, label) in pbar:
                self.train_num += 1
                train_data = to_list(bow.long(), device=self.device)
                bow = bow.to(self.device).float()
                loss, forward_cost, backward_cost, tm_loss, _ = self.model(train_data, bow)

                # for llm refine
                refine_loss = 0
                if self.args.llm_itl and epoch >= self.args.warmStep:
                    # get topics
                    beta = self.model.cal_phi().T
                    top_values, top_idxs = torch.topk(beta, k=self.args.n_topic_words, dim=1)
                    topic_probas = torch.div(top_values, top_values.sum(dim=-1).unsqueeze(-1))
                    topic_words = []
                    for i in range(top_idxs.shape[0]):
                        topic_words.append([self.voc[j] for j in top_idxs[i, :].tolist()])

                    # llm top words and prob
                    suggest_topics, suggest_words = generate_one_pass(llm, tokenizer, topic_words, self.token2idx,
                                                                          instruction_type=self.args.instruction,
                                                                          batch_size=self.args.inference_bs,
                                                                          max_new_tokens=self.args.max_new_tokens)

                    # compute refine loss
                    refine_loss = compute_refine_loss(topic_probas, topic_words, suggest_topics, suggest_words, embedding_model)

                if refine_loss > 0:
                    loss += refine_loss * self.args.refine_weight

                self.optimizer.zero_grad()
                loss.backward()
                for p in self.trainable_params:
                    try:
                        p.grad = p.grad.where(~torch.isnan(p.grad), torch.tensor(0., device=p.grad.device))
                        p.grad = p.grad.where(~torch.isinf(p.grad), torch.tensor(0., device=p.grad.device))
                        nn.utils.clip_grad_norm_(p, max_norm=20, norm_type=2)
                    except:
                        pass
                self.optimizer.step()

                tr_loss.append(loss.item())
                tr_forward_cost.append(forward_cost.item())
                tr_backward_cost.append(backward_cost.item())
                tr_tm.append(tm_loss.item())
                pbar.set_description(f'epoch: {epoch}|{self.epoch}, loss: {np.mean(tr_loss):.4f},  forword_cost: {np.mean(tr_forward_cost):.4f},  '
                                     f'backward_cost: {np.mean(tr_backward_cost):.4f}, TM_loss: {np.mean(tr_tm):.4f}')

            if (epoch + 1) % self.test_every == 0:
                self.model.eval()

                # save tm topics
                topic_dir = 'save_topics/%s' % self.run_name
                if not os.path.exists(topic_dir):
                    os.makedirs(topic_dir)

                tm_topics = []
                phi = self.model.cal_phi().T
                _, top_idxs = torch.topk(phi, k=self.args.n_topic_words, dim=1)
                for i in range(top_idxs.shape[0]):
                    tm_topics.append([self.voc[j] for j in top_idxs[i, :].tolist()])
                with open(os.path.join(topic_dir, 'epoch%s_tm_words.txt' % (epoch + 1)), 'w') as file:
                    for item in tm_topics:
                        file.write(' '.join(item) + '\n')

                # save llm topics
                if self.args.llm_itl and epoch >= self.args.warmStep:
                    llm_topics_dicts, llm_words_dicts = generate_one_pass(llm, tokenizer, tm_topics,
                                                                              self.token2idx,
                                                                              instruction_type=self.args.instruction,
                                                                              batch_size=self.args.inference_bs,
                                                                              max_new_tokens=self.args.max_new_tokens)

                    save_llm_topics(llm_topics_dicts, llm_words_dicts, epoch, topic_dir)

                # save model
                checkpoint_folder = 'save_models/%s' % self.run_name
                if not os.path.exists(checkpoint_folder):
                    os.makedirs(checkpoint_folder)
                torch.save(self.model, os.path.join(checkpoint_folder, 'epoch-%s.pth' % (epoch + 1)))

                self.model.train()
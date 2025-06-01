import sys
sys.path.append("../LLM-ITL")
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from .models.ECRTM import ECRTM
import os
import gensim.downloader as api
from transformers import AutoModelForCausalLM, AutoTokenizer
from topic_models.refine_funcs import compute_refine_loss, save_llm_topics
from  generate import generate_one_pass


class Runner:
    def __init__(self, args, dataset_handler):
        self.args = args
        self.model = ECRTM(args)
        self.dataset_handler = dataset_handler
        self.voc = dataset_handler.vocab
        self.token2idx = {word: index for index, word in enumerate(self.voc)}
        self.run_name = '%s_%s_K%s_seed%s_useLLM-%s' % (args.name, args.dataset, args.n_topic, args.seed, args.llm_itl)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def make_optimizer(self,):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.args.lr,
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def make_lr_scheduler(self, optimizer,):
        lr_scheduler = StepLR(optimizer, step_size=self.args.lr_step_size, gamma=0.5)
        return lr_scheduler

    def train(self, data_loader):
        optimizer = self.make_optimizer()

        if "lr_scheduler" in self.args:
            print("===>Warning: use lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(optimizer)

        data_size = len(data_loader.dataset)

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

        for epoch in range(self.args.epochs):
            self.model.train()
            loss_rst_dict = defaultdict(float)

            for batch_data in data_loader:
                rst_dict = self.model(batch_data)
                batch_loss = rst_dict['loss']

                # for llm refine
                refine_loss = 0
                if self.args.llm_itl and epoch >= self.args.warmStep:
                    # get topics
                    beta = self.model.get_beta()
                    top_values, top_idxs = torch.topk(beta, k=self.args.num_top_word, dim=1)
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
                    batch_loss += refine_loss * self.args.refine_weight

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                for key in rst_dict:
                    loss_rst_dict[key] += rst_dict[key] * len(batch_data)

            if self.args.lr_scheduler:
                lr_scheduler.step()

            output_log = f'Epoch: {epoch:03d}'
            for key in loss_rst_dict:
                output_log += f' {key}: {loss_rst_dict[key] / data_size :.3f}'

            print(output_log)

            # evaluation phase
            if (epoch+1) % self.args.eval_step == 0:
                self.model.eval()

                topic_dir = 'save_topics/%s' % self.run_name
                if not os.path.exists(topic_dir):
                    os.makedirs(topic_dir)

                # save tm topics
                beta = self.model.get_beta().clone().detach().cpu().numpy()
                topic_str_list = self.print_topic_words(beta, self.voc, self.args.num_top_word)
                with open(os.path.join(topic_dir, 'epoch%s_tm_words.txt' % (epoch+1)), 'w') as file:
                    for item in topic_str_list:
                        file.write(' '.join(item.split(' ')) + '\n')

                # save llm topics
                tm_topics = [item.split(' ') for item in topic_str_list]
                if self.args.llm_itl and epoch >= self.args.warmStep:
                    llm_topics_dicts, llm_words_dicts = generate_one_pass(llm, tokenizer, tm_topics, self.token2idx,
                                                                              instruction_type=self.args.instruction,
                                                                              batch_size=self.args.inference_bs,
                                                                              max_new_tokens=self.args.max_new_tokens)

                    save_llm_topics(llm_topics_dicts, llm_words_dicts, epoch, topic_dir)

                # save model
                checkpoint_folder = "save_models/%s" % self.run_name
                if not os.path.exists(checkpoint_folder):
                    os.makedirs(checkpoint_folder)
                torch.save(self.model, os.path.join(checkpoint_folder, 'epoch-%s.pth' % (epoch + 1)))
                self.model.train()

        beta = self.model.get_beta().detach().cpu().numpy()
        return beta


    def test(self, input_data):
        data_size = input_data.shape[0]
        theta = np.zeros((data_size, self.args.n_topic))
        all_idx = torch.split(torch.arange(data_size), self.args.batch_size)

        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_input = input_data[idx]
                batch_theta = self.model.get_theta(batch_input)
                theta[idx] = batch_theta.cpu()

        return theta


    def print_topic_words(self, beta, vocab, num_top_word):
        topic_str_list = list()
        for i, topic_dist in enumerate(beta):
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(num_top_word + 1):-1]
            topic_str = ' '.join(topic_words)
            topic_str_list.append(topic_str)
            #print('Topic {}: {}'.format(i + 1, topic_str))
        return topic_str_list

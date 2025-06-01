'''
Common functions shared by NTMs
'''
import sys
sys.path.append("../LLM-ITL")
import numpy as np
import os
import torch
import ot
from utils import construction_cost


def compute_refine_loss(topic_probas, topic_words, suggest_topics, suggest_words, embedding_model):
    # OT for topic pairs
    n_topic = topic_probas.shape[0]
    ot_dists = torch.zeros(size=(n_topic,)).cuda()
    for i in range(n_topic):
        if suggest_words[i] is not None:
            topic_word_tm = topic_words[i]
            topic_mass_tm = topic_probas[i, :].to(torch.float64)

            topic_word_llm = []
            topic_mass_llm = []
            for k,v in suggest_words[i].items():
                topic_word_llm.append(k)
                topic_mass_llm.append(v)

            cost_M, remove_idx_llm = construction_cost(topic_word_tm, topic_word_llm, embedding_model)

            if len(remove_idx_llm) > 0:
                topic_mass_llm = [topic_mass_llm[i] for i in range(len(topic_mass_llm)) if not i in remove_idx_llm]

            # normalise
            topic_mass_llm = torch.tensor(np.array(topic_mass_llm)).to(torch.float64).cuda()
            topic_mass_llm = topic_mass_llm/topic_mass_llm.sum()

            dist = ot.emd2(topic_mass_tm, topic_mass_llm, cost_M)
            ot_dists[i] = dist

        else:
            ot_dists[i] = 0.

    # get weight for topic ot
    topic_weight = torch.zeros(size=(n_topic,)).cuda()
    for i in range(n_topic):
        if suggest_topics[i] is not None:
            for k, v in suggest_topics[i].items():
                topic_weight[i] = torch.tensor(v).cuda()
        else:
            topic_weight[i] = 0.
    topic_weight = topic_weight.cuda()

    # compute refine loss
    refine_loss = torch.sum(ot_dists * topic_weight)

    return refine_loss


def save_llm_topics(llm_topics_dicts, llm_words_dicts, epoch, log_dir):
    llm_topics = []
    for topic in llm_topics_dicts:
        if topic is not None:
            for k, v in topic.items():
                llm_topics.append(k + ': ' + str(np.round(v, 2)))
        else:
            llm_topics.append('NA')

    with open(os.path.join(log_dir, 'epoch%s_llm_topics.txt' % (epoch + 1)), 'w') as file:
        for item in llm_topics:
            file.write(item + '\n')

    # save llm topic words
    llm_words = []
    for words in llm_words_dicts:
        if words is not None:
            words_topic = []
            for k, v in words.items():
                words_topic.append(k + ': ' + str(np.round(v, 2)))
            llm_words.append(words_topic)
        else:
            llm_words.append(['NA'])

    with open(os.path.join(log_dir, 'epoch%s_llm_words.txt' % (epoch + 1)), 'w') as file:
        for item in llm_words:
            file.write(' '.join(item) + '\n')

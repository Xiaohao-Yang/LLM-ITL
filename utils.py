from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import torch


def sparse2dense(input_matrix):
    if sparse.isspmatrix(input_matrix):
        input_matrix = input_matrix.toarray()
    input_matrix = input_matrix.astype('float32')
    return input_matrix


def get_word_embeddings(words, embedding_model):
    word_embeddings = []
    remove_idxs = []
    for i in range(len(words)):
        try:
            word_embedding = embedding_model[words[i].strip().lower()]
            word_embeddings.append(word_embedding)
        except:
            remove_idxs.append(i)

    return word_embeddings, remove_idxs


def construction_cost(topic_word_tm, topic_word_llm, embedding_model):
    word_embeddings_llm, remove_idxs_llm = get_word_embeddings(topic_word_llm, embedding_model)
    if len(word_embeddings_llm) == 0:
        print('All llm words are removed!')
        print(topic_word_llm)
        quit()

    word_embeddings_tm, remove_idxs_tm = get_word_embeddings(topic_word_tm, embedding_model)
    if len(word_embeddings_tm) == 0:
        print('All topic words are removed!')
        print(topic_word_tm)
        quit()

    # construct cost matrix
    cost_M = 1 - cosine_similarity(word_embeddings_tm, word_embeddings_llm)
    cost_M = torch.from_numpy(cost_M).to(torch.float64).cuda()

    return cost_M, remove_idxs_llm
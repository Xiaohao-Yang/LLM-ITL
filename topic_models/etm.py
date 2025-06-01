import sys
sys.path.append("../LLM-ITL")
import os.path
from topic_models.embedded_topic_model.models.etm import ETM
import argparse
import numpy as np
import gensim.downloader as api
import scipy.io as sio
from utils import sparse2dense
import pickle
from topic_models.hyperparameters import hyperparamters


def format_data_etm(mat_file_name, dataset):
    os.mkdir('datasets/%s' % dataset)
    data = sio.loadmat(mat_file_name)

    train_data = sparse2dense(data['bow_train'])
    test_data = sparse2dense(data['bow_test'])

    train_label = data['label_train']
    test_label = data['label_test']

    voc = data['voc'].reshape(-1).tolist()
    voc = [v[0] for v in voc]

    print('Loading word embedding model...')
    model_glove = api.load("glove-wiki-gigaword-300")
    print('Loading done!')
    word_embeddings = []
    for item in voc:
        word_embeddings.append(model_glove[item])
    word_embeddings = np.array(word_embeddings)

    train_data_dict = {'tokens': [],
                       'counts': [],
                       'labels': train_label}
    for i in range(train_data.shape[0]):
        token_idx = np.where(train_data[i] > 0)[0].astype('int32')
        counts = train_data[i][token_idx].astype('int64')
        train_data_dict['tokens'].append(token_idx)
        train_data_dict['counts'].append(counts)

    test_data_dict = {'tokens': [],
                      'counts': []}

    for i in range(test_data.shape[0]):
        # for original test
        token_idx = np.where(test_data[i] > 0)[0].astype('int32')
        counts = test_data[i][token_idx].astype('int64')
        test_data_dict['tokens'].append(token_idx)
        test_data_dict['counts'].append(counts)

    test_data_all = {'test': test_data_dict,
                     'labels': test_label}

    with open('datasets/%s/train.pkl' % dataset, 'wb') as file:
        pickle.dump(train_data_dict, file)
        file.close()
    with open('datasets/%s/test.pkl' % dataset, 'wb') as file:
        pickle.dump(test_data_all, file)
        file.close()
    with open('datasets/%s/voc.txt' % dataset, 'w') as file:
        file.write(' '.join(voc))
        file.close()
    np.save('datasets/%s/word_embeddings' % dataset, word_embeddings)


def load_data_etm(dataset):
    with open('datasets/%s/voc.txt' % dataset, 'r') as file:
        voc = file.read().split(' ')
        file.close()

    word_embeddings = np.load('datasets/%s/word_embeddings.npy' % dataset)

    with open('datasets/%s/train.pkl' % dataset, 'rb') as file:
        train_data = pickle.load(file)
        file.close()
    with open('datasets/%s/test.pkl' % dataset, 'rb') as file:
        test_data = pickle.load(file)
        file.close()

    return train_data, test_data, word_embeddings, voc


def main():
    if not os.path.exists('datasets/%s' % args.dataset):
        format_data_etm('datasets/%s.mat' % args.dataset, args.dataset)

    train_data, test_data, word_embeddings, voc = load_data_etm(args.dataset)

    print('Loading glove model ...')
    model_glove = api.load("glove-wiki-gigaword-50")
    print('Loading done!')

    etm_instance = ETM(
        voc,
        embeddings=word_embeddings,
        num_topics=args.n_topic,
        epochs=args.n_epochs,
        lr=args.lr,
        seed=args.seed,
        debug_mode=True,
        batch_size=args.bs,
        t_hidden_size=args.hs,
        embedding_model=model_glove
    )
    etm_instance.fit(train_data, test_data, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ETM')
    parser.add_argument("--name", type=str, default="ETM")
    parser.add_argument("--dataset", help="Dataset", default='20News')
    parser.add_argument("--n_topic", help="Number of Topics", default=50, type=int)
    parser.add_argument("--eval_step", default=10, type=int)
    parser.add_argument("--seed", help="Random seed", default=1, type=int)

    parser.add_argument('--warmStep', default=150, type=int)
    parser.add_argument('--llm_itl', action='store_true')
    parser.add_argument('--llm_step', type=int, default=50)  # the number of epochs for llm refine
    parser.add_argument('--llm', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--refine_weight', type=float, default=200)
    parser.add_argument('--instruction', type=str, default='refine_labelTokenProbs',
                        choices=['refine_labelTokenProbs', 'refine_wordIntrusion'])
    parser.add_argument('--inference_bs', type=int, default=100)
    parser.add_argument('--max_new_tokens', type=int, default=300)

    parser.add_argument("--n_epochs", help="Number of epoches", default=250, type=int)
    parser.add_argument("--lr", help="Learning rate", default=0.005, type=float)
    parser.add_argument("--bs", help="Batch size", default=2000, type=int)
    parser.add_argument("--hs", help="Hidden size", default=500, type=int)
    parser.add_argument('--n_topic_words', default=10, type=int)
    args = parser.parse_args()

    # load hyper-parameters for topic model
    hps = hyperparamters[args.name + '_' + args.dataset]
    args.n_epochs = hps[0]
    args.lr = hps[1]
    args.bs = hps[2]

    args.warmStep = args.n_epochs - args.llm_step  # Leave X epochs for LLM refinement

    main()
import sys
sys.path.append("../LLM-ITL")
from optparse import OptionParser
from topic_models.SCHOLAR.scholar import Scholar
from utils import *
import sys
import time
import numpy as np
import gensim
from sklearn.preprocessing import OneHotEncoder
from generate import *
import torch.nn.functional as F
import scipy.io as sio
import gensim.downloader as api
from topic_models.refine_funcs import save_llm_topics
from topic_models.hyperparameters import hyperparamters


def train(model, network_architecture, data_dict, PC, TC, options, test_prior_covars,
          test_topic_covars, batch_size=200, training_epochs=100, init_eta_bn_prop=1.0, rng=None,
          bn_anneal=True, min_weights_sq=1e-7):

    if options.llm_itl:
        # load LLM
        llm = AutoModelForCausalLM.from_pretrained(options.llm,
                                                   trust_remote_code=True,
                                                   torch_dtype=torch.float16
                                                   ).cuda()
        tokenizer = AutoTokenizer.from_pretrained(options.llm, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        print('Loading done!')
    else:
        llm = None
        tokenizer = None

    voc = data_dict['voc']
    token2idx = {word: index for index, word in enumerate(voc)}

    # Train the model
    n_train, vocab_size = data_dict['train_data'].shape

    mb_gen = create_minibatch(data_dict['train_data'], data_dict['train_label'], PC, TC, batch_size=batch_size, rng=rng)
    total_batch = int(n_train / batch_size)

    eta_bn_prop = init_eta_bn_prop  # interpolation between batch norm and no batch norm in final layer of recon

    model.train()

    n_topics = network_architecture['n_topics']
    n_topic_covars = network_architecture['n_topic_covars']
    vocab_size = network_architecture['vocab_size']

    # create matrices to track the current estimates of the priors on the individual weights
    if network_architecture['l1_beta_reg'] > 0:
        l1_beta = 0.5 * np.ones([vocab_size, n_topics], dtype=np.float32) / float(n_train)
    else:
        l1_beta = None
    if network_architecture['l1_beta_c_reg'] > 0 and network_architecture['n_topic_covars'] > 0:
        l1_beta_c = 0.5 * np.ones([vocab_size, n_topic_covars], dtype=np.float32) / float(n_train)
    else:
        l1_beta_c = None
    if network_architecture['l1_beta_ci_reg'] > 0 and network_architecture['n_topic_covars'] > 0 and network_architecture['use_interactions']:
        l1_beta_ci = 0.5 * np.ones([vocab_size, n_topics * n_topic_covars], dtype=np.float32) / float(n_train)
    else:
        l1_beta_ci = None


    # Training cycle
    for epoch in range(training_epochs):
        epoch_start_time = time.time()

        avg_cost = 0.
        avg_nl = 0.
        avg_kld = 0.
        avg_cl = 0.


        # Loop over all batches
        for i in range(total_batch):
            # get a minibatch
            batch_xs, batch_ys, batch_pcs, batch_tcs = next(mb_gen)

            # do one minibatch update
            cost, recon_y, thetas, nl, kld, cl = model.fit(epoch, voc, token2idx, llm, tokenizer, options, batch_xs, batch_ys, batch_pcs, batch_tcs, eta_bn_prop=eta_bn_prop,
                                                           l1_beta=l1_beta, l1_beta_c=l1_beta_c, l1_beta_ci=l1_beta_ci, current_model=options.model)

            # Compute average loss
            avg_cost += float(cost) / n_train * batch_size
            avg_nl += float(nl) / n_train * batch_size
            avg_kld += float(kld) / n_train * batch_size
            avg_cl += float(cl) / batch_size


            if np.isnan(avg_cost):
                print(epoch, i, np.sum(batch_xs, 1).astype(np.int), batch_xs.shape)
                print('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                sys.exit()


        meta = "| epoch {:2d} | time {:5.2f}s ".format(epoch+1, time.time() - epoch_start_time)
        print(meta + "| train loss {:5.2f} | (nll {:4.2f} | kld {:5.2f} | contrastive loss {:5.2f})"
              .format(avg_cost, avg_nl, avg_kld, avg_cl))

        ##########################################################################
        ##########################################################################
        ##########################################################################
        ##########################################################################
        # if we're using regularization, update the priors on the individual weights
        if network_architecture['l1_beta_reg'] > 0:
            weights = model.get_weights().T
            weights_sq = weights ** 2
            # avoid infinite regularization
            weights_sq[weights_sq < min_weights_sq] = min_weights_sq
            l1_beta = 0.5 / weights_sq / float(n_train)

        if network_architecture['l1_beta_c_reg'] > 0 and network_architecture['n_topic_covars'] > 0:
            weights = model.get_covar_weights().T
            weights_sq = weights ** 2
            weights_sq[weights_sq < min_weights_sq] = min_weights_sq
            l1_beta_c = 0.5 / weights_sq / float(n_train)

        if network_architecture['l1_beta_ci_reg'] > 0 and network_architecture['n_topic_covars'] > 0 and network_architecture['use_interactions']:
            weights = model.get_covar_interaction_weights().T
            weights_sq = weights ** 2
            weights_sq[weights_sq < min_weights_sq] = min_weights_sq
            l1_beta_ci = 0.5 / weights_sq / float(n_train)

        # anneal eta_bn_prop from 1.0 to 0.0 over training
        if bn_anneal:
            if eta_bn_prop > 0:
                eta_bn_prop -= 1.0 / float(0.75 * training_epochs)
                if eta_bn_prop < 0:
                    eta_bn_prop = 0.0

        if (epoch + 1) % options.eval_step == 0:
            model.eval()

            run_name = '%s_%s_K%s_seed%s_useLLM-%s' % (options.model, options.dataset, options.n_topic, options.seed, options.llm_itl)

            topic_dir = 'save_topics/%s' % run_name
            checkpoint_folder = "save_models/%s" % run_name
            if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder)
            if not os.path.exists(topic_dir):
                os.makedirs(topic_dir)

            # save tm topics
            beta = model._model.beta_layer.weight.T.to(torch.float64)
            beta = F.softmax(beta, dim=1)
            _, top_idxs = torch.topk(beta, k=options.n_topic_words, dim=1)
            tm_topics = []
            for i in range(top_idxs.shape[0]):
                tm_topics.append([voc[j] for j in top_idxs[i, :].tolist()])

            with open(os.path.join(topic_dir, 'epoch%s_tm_words.txt' % (epoch + 1)), 'w') as file:
                for item in tm_topics:
                    file.write(' '.join(item) + '\n')

            # save llm topics
            if options.llm_itl and epoch >= options.warmStep:
                llm_topics_dicts, llm_words_dicts = generate_one_pass(llm, tokenizer, tm_topics, token2idx,
                                                                          instruction_type=options.instruction,
                                                                          batch_size=options.inference_bs,
                                                                          max_new_tokens=options.max_new_tokens)

                save_llm_topics(llm_topics_dicts, llm_words_dicts, epoch, topic_dir)
            # save model
            torch.save(model, os.path.join(checkpoint_folder, 'epoch-%s.pth' % (epoch + 1)))

            model.train()

    # finish training
    model.eval()
    return model


def get_init_bg(data):
    #Compute the log background frequency of all words
    #sums = np.sum(data, axis=0)+1
    n_items, vocab_size = data.shape
    sums = np.array(data.sum(axis=0)).reshape((vocab_size,))+1.
    print("Computing background frequencies")
    print("Min/max word counts in training data: %d %d" % (int(np.min(sums)), int(np.max(sums))))
    bg = np.array(np.log(sums) - np.log(float(np.sum(sums))), dtype=np.float32)
    return bg


def load_word_vectors(options, rng, vocab):
    # load word2vec vectors if given
    if options.word2vec_file is not None:
        vocab_size = len(vocab)
        vocab_dict = dict(zip(vocab, range(vocab_size)))
        # randomly initialize word vectors for each term in the vocabualry
        embeddings = np.array(rng.rand(options.emb_dim, vocab_size) * 0.25 - 0.5, dtype=np.float32)
        count = 0
        print("Loading word vectors")
        # load the word2vec vectors
        pretrained = gensim.models.KeyedVectors.load_word2vec_format(options.word2vec_file, binary=True)

        # replace the randomly initialized vectors with the word2vec ones for any that are available
        for word, index in vocab_dict.items():
            if word in pretrained:
                count += 1
                embeddings[:, index] = pretrained[word]

        print("Found embeddings for %d words" % count)
        update_embeddings = False
    else:
        embeddings = None
        update_embeddings = True

    return embeddings, update_embeddings


def make_network(options, vocab_size, label_type=None, n_labels=0, n_prior_covars=0, n_topic_covars=0):
    # Assemble the network configuration parameters into a dictionary
    network_architecture = \
        dict(embedding_dim=options.emb_dim,
             n_topics=options.n_topic,
             vocab_size=vocab_size,
             #word_embedding=word_embedding,
             label_type=label_type,
             n_labels=n_labels,
             n_prior_covars=n_prior_covars,
             n_topic_covars=n_topic_covars,
             l1_beta_reg=options.l1_topics,
             l1_beta_c_reg=options.l1_topic_covars,
             l1_beta_ci_reg=options.l1_interactions,
             l2_prior_reg=options.l2_prior_covars,
             classifier_layers=1,
             use_interactions=options.interactions,
             dist=options.dist,
             model=options.model
             )
    return network_architecture


def create_minibatch(X, Y, PC, TC, batch_size=200, rng=None):
    # Yield a random minibatch
    while True:
        # Return random data samples of a size 'minibatch_size' at each iteration
        if rng is not None:
            ixs = rng.randint(X.shape[0], size=batch_size)
        else:
            ixs = np.random.randint(X.shape[0], size=batch_size)

        X_mb = np.array(X[ixs, :].todense()).astype('float32')
        if Y is not None:
            Y_mb = Y[ixs, :].astype('float32')
        else:
            Y_mb = None

        if PC is not None:
            PC_mb = PC[ixs, :].astype('float32')
        else:
            PC_mb = None

        if TC is not None:
            TC_mb = TC[ixs, :].astype('float32')
        else:
            TC_mb = None

        yield X_mb, Y_mb, PC_mb, TC_mb


def split_matrix(train_X, train_indices, dev_indices):
    # split a matrix (word counts, labels, or covariates), into train and dev
    if train_X is not None and dev_indices is not None:
        dev_X = train_X[dev_indices, :]
        train_X = train_X[train_indices, :]
        return train_X, dev_X
    else:
        return train_X, None


def train_dev_split(options, rng):
    # randomly split into train and dev
    if options.dev_folds > 0:
        n_dev = int(options.n_train / options.dev_folds)
        indices = np.array(range(options.n_train), dtype=int)
        rng.shuffle(indices)
        if options.dev_fold < options.dev_folds - 1:
            dev_indices = indices[n_dev * options.dev_fold: n_dev * (options.dev_fold +1)]
        else:
            dev_indices = indices[n_dev * options.dev_fold:]
        train_indices = list(set(indices) - set(dev_indices))
        return train_indices, dev_indices

    else:
        return None, None


def load_data_scholar(mat_file_name):
    data = sio.loadmat(mat_file_name)
    train_data = data['bow_train']
    test_data = data['bow_test']
    voc = data['voc'].transpose()
    voc = [v[0][0] for v in voc]

    train_label = data['label_train']
    test_label = data['label_test']

    # change label to one-hot vector
    label_encoder = OneHotEncoder()
    train_label_oneHot = label_encoder.fit_transform(train_label).todense()
    test_label_oneHot = label_encoder.transform(test_label).todense()

    if not sparse.isspmatrix(train_data):
        train_data = sparse.csr_matrix(train_data).astype('float32')
        test_data = sparse.csr_matrix(test_data).astype('float32')

    data_dict = {
        'train_data': train_data,
        'train_label': train_label_oneHot,
        'train_label_normal': train_label,
        'test_data': test_data,
        'test_label': test_label_oneHot,
        'test_label_normal': test_label,
        'voc': voc,
    }

    return data_dict


def main(data_dict, model_glove):
    if options.r:
        options.l1_topics = 1.0
        options.l1_topic_covars = 1.0
        options.l1_interactions = 1.0
    if options.seed is not None:
        rng = np.random.RandomState(options.seed)
        seed = options.seed
    else:
        rng = np.random.RandomState(np.random.randint(0, 100000))
        seed = None

    n_labels = data_dict['train_label'].shape[1]
    options.n_train, vocab_size = data_dict['train_data'].shape
    options.n_labels = n_labels

    # variables in original implementation
    train_prior_covars, prior_covar_selector, prior_covar_names, n_prior_covars = None, None, None, 0
    train_topic_covars, topic_covar_selector, topic_covar_names, n_topic_covars = None, None, None, 0
    test_prior_covars = None
    test_topic_covars = None
    label_type = None

    train_indices, dev_indices = train_dev_split(options, rng)
    train_X, dev_X = split_matrix(data_dict['train_data'], train_indices, dev_indices)
    train_prior_covars, dev_prior_covars = split_matrix(train_prior_covars, train_indices, dev_indices)
    train_topic_covars, dev_topic_covars = split_matrix(train_topic_covars, train_indices, dev_indices)
    n_train, _ = train_X.shape
    # initialize the background using overall word frequencies
    init_bg = get_init_bg(train_X)
    if options.no_bg:
        init_bg = np.zeros_like(init_bg)
    # load word vectors
    embeddings, update_embeddings = load_word_vectors(options, rng, data_dict['voc'])
    #################################################################################
    #################################################################################

    network_architecture = make_network(options, vocab_size, label_type, n_labels, n_prior_covars,
                                        n_topic_covars)
    print("Network architecture:")
    for key, val in network_architecture.items():
        print(key + ':', val)

    # create the model
    model = Scholar(network_architecture, model_glove, alpha=options.alpha, learning_rate=options.lr,
                    init_embeddings=embeddings, update_embeddings=update_embeddings, init_bg=init_bg,
                    adam_beta1=options.momentum, device=options.device, seed=seed,
                    classify_from_covars=options.covars_predict, model=options.model, topk=options.topk)

    # train the model
    print("Optimizing full model")
    model = train(model, network_architecture, data_dict, train_prior_covars, train_topic_covars, options,
                  test_prior_covars, test_topic_covars, training_epochs=options.n_epochs,
                  batch_size=options.bs, rng=rng)


if __name__ == '__main__':
    args = sys.argv[1:]
    usage = "%prog input_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--model', type=str, default='scholar')
    parser.add_option('--dataset', type=str, default='20News')
    parser.add_option('--n_topic', type=int, default=50)
    parser.add_option('--seed', type=int, default=1, help='Random seed: default=%default')
    parser.add_option('--eval_step', type=int, default=10)

    parser.add_option('--n_epochs', type=int, default=550)
    parser.add_option('--warmStep', default=450, type=int)
    parser.add_option('--llm_itl', action='store_true', default=False)
    parser.add_option('--llm_step', type=int, default=50)  # the number of epochs for llm refine
    parser.add_option('--llm', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_option('--refine_weight', type=float, default=200)
    parser.add_option('--instruction', type=str, default='refine_labelTokenProbs') # choices=['refine_labelTokenProbs', 'refine_wordIntrusion']
    parser.add_option('--inference_bs', type=int, default=5)
    parser.add_option('--max_new_tokens', type=int, default=300)
    parser.add_option('--n_topic_words', default=10, type=int)

    parser.add_option('--lr', type=float, default=0.002)
    parser.add_option('--bs', type=int, default=200, help='Size of minibatches: default=%default')

    parser.add_option('--topk', type=int, default=15)
    parser.add_option('-m', dest='momentum', type=float, default=0.99,
                      help='beta1 for Adam: default=%default')
    parser.add_option('--train-prefix', type=str, default='train',
                      help='Prefix of train set: default=%default')
    parser.add_option('--test-prefix', type=str, default='test',
                      help='Prefix of test set: default=%default')
    parser.add_option('--labels', type=str, default=False,
                      help='Read labels from input_dir/[train|test].labels.csv: default=%default')
    parser.add_option('--prior-covars', type=str, default=None,
                      help='Read prior covariates from files with these names (comma-separated): default=%default')
    parser.add_option('--topic-covars', type=str, default=None,
                      help='Read topic covariates from files with these names (comma-separated): default=%default')
    parser.add_option('--interactions', action="store_true", default=False,
                      help='Use interactions between topics and topic covariates: default=%default')
    parser.add_option('--covars-predict', action="store_true", default=False,
                      help='Use covariates as input to classifier: default=%default')
    parser.add_option('--min-prior-covar-count', type=int, default=None,
                      help='Drop prior covariates with less than this many non-zero values in the training dataa: default=%default')
    parser.add_option('--min-topic-covar-count', type=int, default=None,
                      help='Drop topic covariates with less than this many non-zero values in the training dataa: default=%default')
    parser.add_option('-r', action="store_true", default=False,
                      help='Use default regularization: default=%default')
    parser.add_option('--l1-topics', type=float, default=0.0,
                      help='Regularization strength on topic weights: default=%default')
    parser.add_option('--l1-topic-covars', type=float, default=0.0,
                      help='Regularization strength on topic covariate weights: default=%default')
    parser.add_option('--l1-interactions', type=float, default=0.0,
                      help='Regularization strength on topic covariate interaction weights: default=%default')
    parser.add_option('--l2-prior-covars', type=float, default=0.0,
                      help='Regularization strength on prior covariate weights: default=%default')
    parser.add_option('--o', dest='output_dir', type=str, default='output',
                      help='Output directory: default=%default')
    parser.add_option('--emb-dim', type=int, default=300,
                      help='Dimension of input embeddings: default=%default')
    parser.add_option('--w2v', dest='word2vec_file', type=str, default=None,
                      help='Use this word2vec .bin file to initialize and fix embeddings: default=%default')
    parser.add_option('--alpha', type=float, default=1.0,
                      help='Hyperparameter for logistic normal prior: default=%default')
    parser.add_option('--no-bg', action="store_true", default=False,
                      help='Do not use background freq: default=%default')
    parser.add_option('--dev-folds', type=int, default=0,
                      help='Number of dev folds: default=%default')
    parser.add_option('--dev-fold', type=int, default=0,
                      help='Fold to use as dev (if dev_folds > 0): default=%default')
    parser.add_option('--device', type=int, default=0,
                      help='GPU to use: default=%default')
    parser.add_option('--dist', type=int, default=0, help='distance')
    options, args = parser.parse_args(args)

    torch.manual_seed(options.seed)
    np.random.seed(options.seed)

    # load hyper-parameters for topic model
    hps = hyperparamters[options.model + '_' + options.dataset]
    options.n_epochs = hps[0]
    options.lr = hps[1]
    options.bs = hps[2]

    options.warmStep = options.n_epochs - options.llm_step  # Leave X epochs for LLM refinement

    # load data
    data_dict = load_data_scholar('datasets/%s.mat' % options.dataset)

    print('Loading glove model ...')
    model_glove = api.load("glove-wiki-gigaword-50")
    print('Loading done!')

    main(data_dict, model_glove)

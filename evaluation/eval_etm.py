import sys
sys.path.append("../LLM-ITL")
import argparse
import os
import torch
from metrics import *
import json
from tqdm import tqdm
from topic_models.etm import load_data_etm
from topic_models.embedded_topic_model.models.etm import ETM
from topic_models.embedded_topic_model.utils import data as Data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument("--model_folder", type=str, required=True)
    parser.add_argument('--dataset', type=str, default='20News')
    parser.add_argument("--inference_bs", type=int, default=500)
    parser.add_argument("--eval_topics", action='store_true')
    parser.add_argument('--TC', type=str, default='cv')
    args = parser.parse_args()
    device = 'cuda'

    checkpoint_folder = 'save_models'
    model_epochs = os.listdir(os.path.join(checkpoint_folder, args.model_folder))
    model_epochs = sorted(model_epochs, key=lambda x: int(x.split('-')[1].split('.')[0]))

    # load data
    train_data, test_data, _, voc = load_data_etm(args.dataset)

    test_label = test_data['labels'].reshape(-1, )
    num_docs_test = len(test_data['test']['counts'])

    train_label = train_data['labels'].reshape(-1, )
    num_docs_train = len(train_data['counts'])

    vocabulary_size = len(voc)

    # save path
    save_path = 'evaluation_output/%s' % (args.model_folder + '.jsonl')
    open(save_path, 'w').close()

    for model_epoch in tqdm(model_epochs):
        # load model
        model = torch.load(os.path.join(checkpoint_folder, args.model_folder, model_epoch), weights_only=False)
        model.eval()

        # infer topic proportion for test
        with torch.no_grad():
            indices = torch.tensor(range(num_docs_test))
            indices = torch.split(indices, args.inference_bs)
            thetas = []
            for idx, ind in enumerate(indices):
                data_batch = Data.get_batch(
                    test_data['test']['tokens'],
                    test_data['test']['counts'],
                    ind,
                    vocabulary_size,
                    device)
                sums = data_batch.sum(1).unsqueeze(1)
                normalized_data_batch = data_batch / sums
                theta, _ = model.get_theta(normalized_data_batch)
                thetas.append(theta)
            doc_topic = torch.cat(tuple(thetas), 0).cpu().numpy()

        # infer topic proportion for train
        with torch.no_grad():
            indices = torch.tensor(range(num_docs_train))
            indices = torch.split(indices, args.inference_bs)
            thetas = []
            for idx, ind in enumerate(indices):
                data_batch = Data.get_batch(
                    train_data['tokens'],
                    train_data['counts'],
                    ind,
                    vocabulary_size,
                    device)
                sums = data_batch.sum(1).unsqueeze(1)
                normalized_data_batch = data_batch / sums
                theta, _ = model.get_theta(normalized_data_batch)
                thetas.append(theta)
            doc_topic_train = torch.cat(tuple(thetas), 0).cpu().numpy()

        ############## evaluate theta ##############
        TP, TN = compute_TP_TN(test_label, doc_topic)
        PN = (TP + TN)/2

        # more available metrics
        '''
        # doc cls acc
        acc = rf_cls(doc_topic_train, train_label, doc_topic, test_label)

        # metrics used by TopicGPT
        pred = np.argmax(doc_topic, axis=1)
        purity, inverse_purity, harmonic_purity = calculate_purity(test_label, pred)

        ari = metrics.adjusted_rand_score(test_label, pred)
        mis = metrics.normalized_mutual_info_score(test_label, pred)
        '''
        ############## evaluate topics ##############
        if args.eval_topics:
            print('Evaluating Topic Coherence (Metrics: %s) ...' % args.TC)
            topic_file = 'save_topics/%s/epoch%s_tm_words.txt' % (args.model_folder, model_epoch.split('.')[0].split('-')[1])
            temp_folder = 'temp_tc/%s/' % args.model_folder
            temp_path = os.path.join(temp_folder, 'epoch%s_tm_words_%s.txt' % (model_epoch.split('.')[0].split('-')[1], args.TC))
            if not os.path.exists(temp_folder):
                os.makedirs(temp_folder)
            open(temp_path, 'w').close()

            if args.TC == 'cv':
                tc_metric = 'C_V'
            os.system('java -jar palmetto-0.1.5-exec.jar wikipedia_bd %s %s > %s' % (tc_metric, topic_file, temp_path))

            with open(temp_path, 'r') as file:
                content = file.read()
                tc_agg = read_tc(content)

            # eval topics diversity
            with open(topic_file, 'r') as file:
                topics = file.readlines()
                topics = [t.strip().split(' ') for t in topics]

            td = topic_diversity(topics)
        else:
            tc_agg = None
            td = None

        ############## save metrics ############
        opt_dict = {'TP': TP,
                    'TN': TN,
                    'PN': PN,
                    'TC': tc_agg,
                    'TD': td
                    }
        with open(save_path, 'a') as f:
            f.write(json.dumps(opt_dict) + '\n')


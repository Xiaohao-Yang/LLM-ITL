import sys
sys.path.append("../LLM-ITL")
import argparse
import os
from topic_models.ECRTM.utils.data.TextData import TextData
from topic_models.ECRTM.models import ECRTM
import torch
from metrics import *
import json
from tqdm import tqdm



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument("--model_folder", type=str, required=True)
    parser.add_argument('--dataset', type=str, default='20News')
    parser.add_argument("--inference_bs", type=int, default=500)
    parser.add_argument("--eval_topics", action='store_true')
    parser.add_argument('--TC', type=str, default='cv')
    args = parser.parse_args()

    checkpoint_folder = 'save_models'
    model_epochs = os.listdir(os.path.join(checkpoint_folder, args.model_folder))
    model_epochs = sorted(model_epochs, key=lambda x: int(x.split('-')[1].split('.')[0]))

    # load data
    dataset_handler = TextData(args.dataset, args.inference_bs)

    train_data = dataset_handler.train_data
    train_label = dataset_handler.train_labels

    test_data = dataset_handler.test_data
    test_label = dataset_handler.test_labels

    # save path
    save_path = 'evaluation_output/%s' % (args.model_folder + '.jsonl')
    open(save_path, 'w').close()

    for model_epoch in tqdm(model_epochs):
        # load model
        model = torch.load(os.path.join(checkpoint_folder, args.model_folder, model_epoch), weights_only=False)
        model.eval()

        # infer topic proportion for test
        with torch.no_grad():
            doc_topic = model.get_theta(test_data).cpu().numpy()
            doc_topic_train = model.get_theta(train_data).cpu().numpy()

        ############## evaluate theta ##############
        TP, TN = compute_TP_TN(test_label, doc_topic)
        PN = (TP + TN) / 2

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

            # make sure you have unzip 'Wikipedia_bd.zip' before this
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



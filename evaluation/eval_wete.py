import sys
sys.path.append("../LLM-ITL")
import argparse
import os
from topic_models.WeTe.dataloader import dataloader
from topic_models.WeTe.model import WeTe
from topic_models.WeTe.Utils import to_list
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

    device = 'cuda'

    checkpoint_folder = 'save_models'
    model_epochs = os.listdir(os.path.join(checkpoint_folder, args.model_folder))
    model_epochs = sorted(model_epochs, key=lambda x: int(x.split('-')[1].split('.')[0]))

    # load data
    train_loader, _ = dataloader(dataname=args.dataset, mode='train', batch_size=args.inference_bs)
    test_loader, _ = dataloader(dataname=args.dataset, mode='test', batch_size=args.inference_bs)

    # train_label = train_loader.dataset.label
    # test_label = test_loader.dataset.label

    # save path
    save_path = 'evaluation_output/%s' % (args.model_folder + '.jsonl')
    open(save_path, 'w').close()

    for model_epoch in tqdm(model_epochs):
        # load model
        model = torch.load(os.path.join(checkpoint_folder, args.model_folder, model_epoch), weights_only=False)
        model.eval()

        # infer test doc-topic
        test_theta = None
        test_label = None
        with torch.no_grad():
            for bow, label in test_loader:
                test_data = to_list(bow.long(), device=device)
                bow = bow.to(device).float()
                _, _, _, _, theta = model(test_data, bow)

                if test_theta is None:
                    test_theta = theta.cpu().numpy()
                    test_label = label.numpy()
                else:
                    test_theta = np.concatenate((test_theta, theta.cpu().numpy()))
                    test_label = np.concatenate((test_label, label.numpy()))

        # infer train doc-topic
        train_theta = None
        train_label = None
        with torch.no_grad():
            for bow, label in train_loader:
                train_data = to_list(bow.long(), device=device)
                bow = bow.to(device).float()
                _, _, _, _, theta = model(train_data, bow)

                if train_theta is None:
                    train_theta = theta.cpu().numpy()
                    train_label = label.numpy()
                else:
                    train_theta = np.concatenate((train_theta, theta.cpu().numpy()))
                    train_label = np.concatenate((train_label, label.numpy()))

        ############## evaluate theta ##############
        doc_topic = test_theta
        doc_topic_train = train_theta

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
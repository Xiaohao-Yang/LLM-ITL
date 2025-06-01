import sys
sys.path.append("../LLM-ITL")
import numpy as np
import yaml
import argparse
from topic_models.ECRTM.Runner import Runner
from topic_models.ECRTM.utils.data.TextData import TextData
import torch
from topic_models.hyperparameters import hyperparamters


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="ECRTM")
    parser.add_argument("--dataset", type=str, default='20News')
    parser.add_argument('--n_topic', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument("--eval_step", default=10, type=int)
    parser.add_argument('--batch_size', type=int, default=200)

    parser.add_argument('--sinkhorn_alpha', type=float, default=20)
    parser.add_argument('--OT_max_iter', type=int, default=1000)
    parser.add_argument('--weight_loss_ECR', type=float, default=250)

    # llm
    parser.add_argument('--warmStep', default=0, type=int)
    parser.add_argument('--llm_itl', action='store_true')
    parser.add_argument('--llm', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--llm_step', type=int, default=50)  # the number of epochs for llm refine
    parser.add_argument('--refine_weight', type=float, default=200)
    parser.add_argument('--instruction', type=str, default='refine_labelTokenProbs',
                        choices=['refine_labelTokenProbs', 'refine_wordIntrusion'])
    parser.add_argument('--inference_bs', type=int, default=5)
    parser.add_argument('--max_new_tokens', type=int, default=300)

    parser.add_argument('--lr_scheduler', type=bool, default=False)
    parser.add_argument('--lr_step_size', type=int, default=125)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--en1_units', type=int, default=200)
    parser.add_argument('--beta_temp', type=float, default=0.2)
    parser.add_argument('--num_top_word', type=int, default=10)
    args = parser.parse_args()

    # load hyper-parameters for topic model
    hps = hyperparamters[args.name + '_' + args.dataset]
    args.epochs = hps[0]
    args.lr = hps[1]
    args.batch_size = hps[2]
    args.weight_loss_ECR = hps[3]

    args.warmStep = args.epochs - args.llm_step # Leave X epochs for LLM refinement

    return args



def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    seperate_line_log = '=' * 70
    print(seperate_line_log)
    print(seperate_line_log)
    print('\n' + yaml.dump(vars(args), default_flow_style=False))

    dataset_handler = TextData(args.dataset, args.batch_size)

    args.vocab_size = dataset_handler.train_data.shape[1]
    args.word_embeddings = dataset_handler.word_embeddings

    # train model via runner.
    runner = Runner(args, dataset_handler)

    beta = runner.train(dataset_handler.train_loader)


if __name__ == '__main__':
    main()

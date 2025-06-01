import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='20News')
parser.add_argument('--model', type=str, default='nvdm',
                    choices=['nvdm', 'plda', 'nstm', 'etm', 'scholar', 'clntm', 'wete', 'ecrtm'])
parser.add_argument('--n_topic', type=int, default=50)
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--eval_step', type=int, default=2)

parser.add_argument('--llm_itl', action='store_true', help='Use LLM or not')
parser.add_argument('--inference_bs', type=int, default=100) # set this number based on your GPU memory
parser.add_argument('--llm_step', type=int, default=5) # the number of epochs for llm refine
args = parser.parse_args()


if __name__ == '__main__':
    if not args.model in ['scholar', 'clntm']:
        argument = ('python topic_models/%s.py --dataset=%s --n_topic=%s --seed=%s --eval_step=%s --inference_bs=%s --llm_step=%s' %
                    (args.model, args.dataset, args.n_topic, args.random_seed, args.eval_step, args.inference_bs, args.llm_step))

    else:
        argument = ('python topic_models/scholar.py --model=%s --dataset=%s --n_topic=%s --seed=%s --eval_step=%s --inference_bs=%s --llm_step=%s' %
                    (args.model, args.dataset, args.n_topic, args.random_seed, args.eval_step, args.inference_bs, args.llm_step))

    argument += ' --llm_itl' if args.llm_itl else ''
    os.system(argument)

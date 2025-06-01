import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='20News')
parser.add_argument('--model', type=str, default='nvdm',
                    choices=['nvdm', 'plda', 'nstm', 'etm', 'scholar', 'clntm', 'wete', 'ecrtm'])
parser.add_argument('--n_topic', type=int, default=50)
parser.add_argument('--eval_topics', action='store_true')
parser.add_argument('--llm_itl', action='store_true')
args = parser.parse_args()


if __name__ == '__main__':
    model_folders = os.listdir('save_models')
    model_folders = [f for f in model_folders if ('%s_%s_K%s' %
                     (args.model, args.dataset, args.n_topic)).lower() in f.lower()
                     and str(args.llm_itl).lower() in f.lower()]
    print('Evaluation for:' )
    print(model_folders)

    '''
    add more filter conditions if needed
    '''

    for model_folder in model_folders:
        if args.model in ['scholar', 'clntm']:
            argument = ('python evaluation/eval_scholar.py --model_folder=%s --dataset=%s' %
                        (model_folder, args.dataset))
        else:
            argument = ('python evaluation/eval_%s.py --model_folder=%s --dataset=%s' %
                            (args.model, model_folder, args.dataset))

        argument += ' --eval_topics' if args.eval_topics else ''
        os.system(argument)



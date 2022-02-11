# coding: UTF-8
import time
import os
import torch
import numpy as np
from importlib import import_module
import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser(description='Fine-grained Ethical Behavior Discrimination')
parser.add_argument('--task', type=str, default='BE', help='choose a task: BE or PR')
# task_Behavior-Extraction : #ERNIE-OT#、#BERT-OT#、ERNIE_CRF、BERT_CRF
# task_Purpose-Reasoning : ERNIE_LSTM、ERNIE、BERT、BERT_LSTM
parser.add_argument('--model', default='BERT_CRF', type=str, help='choose a model: ERNIE, ERNIE_LSTM, ERNIE_CRF, BERT_CRF')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--tagging_schema', type=str, default="BIO", help="OT/BIO")
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'data'  # 数据集

    model_name = args.model
    if args.task == 'BE':
        x = import_module('BE-models.' + model_name)
        # task_Behavior-Extraction : ERNIE、BERT、ERNIE_CRF、BERT_CRF
        from crf_utils import build_dataset, build_iterator, get_time_dif
        if 'CRF' in model_name:
            from crf_train_eval import train
        else:
            from ot_train_eval import train
    else:
        x = import_module('PR-models.' + model_name)
        # task_Purpose-Reasoning : ERNIE_LSTM、ERNIE、BERT、BERT_LSTM
        from utils import build_dataset, build_iterator, get_time_dif
        from train_eval import train
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print(model_name)
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config, args.tagging_schema)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)

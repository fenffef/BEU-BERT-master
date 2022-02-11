# coding: UTF-8
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'ERNIE'
        self.train_path = dataset + '/pr/train.txt'                                # 训练集
        self.dev_path = dataset + '/pr/dev.txt'                                  # 验证集
        self.test_path = dataset + '/pr/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/pr/class.txt', encoding='utf-8').readlines()]
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 10                                             # epoch数
        self.batch_size = 16                                           # mini-batch大小
        self.pad_size = 100                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 2e-5                                      # 学习率
        self.bert_path = './ERNIE_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.dropout = 0.1


class LocationEncoding(nn.Module):
    def __init__(self):
        super(LocationEncoding, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x, pos_inx):
        batch_size, seq_len = x.size()[0], x.size()[1]
        x = pos_inx.to(self.device).unsqueeze(2) * x
        return x


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.location = LocationEncoding()
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x, labels):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        beh_weight = x[3]
        pur_weight = x[4]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.dropout(encoder_out)

        beh_out = self.location(out, beh_weight)
        beh_len = torch.sum(beh_weight, dim=1)
        beh_pool = torch.sum(beh_out, dim=1)
        beh_pool = torch.div(beh_pool, beh_len.float().unsqueeze(-1)).float()

        pur_out = self.location(out, pur_weight)
        pur_len = torch.sum(pur_weight, dim=1)
        pur_pool = torch.sum(pur_out, dim=1)
        pur_pool = torch.div(pur_pool, pur_len.float().unsqueeze(-1)).float()

        beh_pur = torch.cat([beh_pool, pur_pool], 1)
        scores = self.fc(beh_pur)
        return scores

# coding: UTF-8
import torch
torch.cuda.empty_cache()
import torch.nn as nn
from torchcrf import CRF
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.bert_path = './chinese-xlnet-base'
        # bert-pretrain / chinese-roberta-wwm-ext / chinese-xlnet-base / ernie-pretrain / roberta-base
        self.model_name = 'BERT_CRF'
        self.train_path = dataset + '/ote/train.txt'                                # 训练集
        self.dev_path = dataset + '/ote/dev.txt'                                  # 验证集
        self.test_path = dataset + '/ote/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/ote/class.txt', encoding='utf-8').readlines()]
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 10                                            # epoch数
        self.batch_size = 8                                            # mini-batch大小
        self.pad_size = 100                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 2e-5                                       # 学习率
        self.num_ner_labels = len(self.class_list)                      # 类别数
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.dropout = 0.1
        self.rnn_hidden = 768
        self.num_layers = 2


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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for param in self.bert.parameters():
            param.requires_grad = True
        self.location = LocationEncoding()
        self.num_ner_labels = config.num_ner_labels
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.CRF_model = CRF(config.num_ner_labels, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.rnn_hidden, config.num_ner_labels)

    def forward(self, x, labels):
        context = x[0]  # 输入的句子
        lengths = x[1]
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        ner_masks = x[3]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        encoder_out = self.dropout(encoder_out)
        emission = self.fc(encoder_out)

        '''
        emissions (`~torch.Tensor`): Emission score tensor of size
        ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
        ``(batch_size, seq_length, num_tags)`` otherwise.
        '''
        ner_loss_list = self.CRF_model(emission, labels, mask.type(torch.ByteTensor).to(self.device), reduction='none')
        ner_loss = torch.mean(-ner_loss_list)
        ner_predict = self.CRF_model.decode(emission, ner_masks.type(torch.ByteTensor).to(self.device))

        return ner_loss, ner_predict, lengths

#coding:utf-8
import time
import torch
import numpy as np
#from train_eval import train, init_network
#from importlib import import_module
from models import bert
#import argparse
#from utils import build_dataset, build_iterator, get_time_dif



PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号
def get_ids(text,config):
    pad_size=config.pad_size
    token = config.tokenizer.tokenize(text)
    token = [CLS] + token
    seq_len = len(token)
    mask = []
    token_ids = config.tokenizer.convert_tokens_to_ids(token)

    if pad_size:
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
            token_ids += ([0] * (pad_size - len(token)))
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            seq_len = pad_size
    x = torch.LongTensor([token_ids]).to(config.device)
    # pad前的长度(超过pad_size的设为pad_size)
    print(pad_size,seq_len)
    seq_len = torch.LongTensor([seq_len]).to(config.device)
    mask = torch.LongTensor([mask]).to(config.device)
    return (x, seq_len, mask)


#text="杨澜公布赴台采访老照片 双颊圆润似董洁(组图)"
text="刘嘉玲称工作令其容光焕发"


if True:
    dataset = 'THUCNews'  # 数据集
    #x = import_module('models.bert')
    x = bert
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    #torch.cuda.manual_seed_all(1)
    #torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    #train_data, dev_data, test_data = build_dataset(config)
    #test_iter = build_iterator(test_data, config)

    model = x.Model(config).to(config.device)

    model.load_state_dict(torch.load(config.save_path,map_location='cpu'))
    model.eval()
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        texts = get_ids(text,config)
        print(texts)
        outputs = model(texts)
        predic = torch.max(outputs.data, 1)[1].cpu().numpy()
        predict_all = np.append(predict_all, predic)
    print(predict_all)




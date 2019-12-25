
import time
import torch
import numpy as np
#from train_eval import train, init_network
#from importlib import import_module
import bert
import time
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
    #print(pad_size,seq_len)
    seq_len = torch.LongTensor([seq_len]).to(config.device)
    mask = torch.LongTensor([mask]).to(config.device)
    return (x, seq_len, mask)


#text="杨澜公布赴台采访老照片 双颊圆润似董洁(组图)"
#text="施工人员未佩戴安全带"
#text="其他 5DA  14.6m板未按要求进行孔洞封堵以及防漏雨措施施工"
#text="未设置防护围栏或设置不符合规范要求 综合实验楼4区5区6区物料提升机内侧防护门未设置，缓冲底座未设置，安全通道及操作台防护棚未及时设置，吊笼与主体防护架二次结构工人私自搭设，6区外侧防护门缺失 5区操作台升降按钮缺失未及时维修"
#text="用电设备未有各自专用的开关箱 用电设备未有各自开关箱箱内接地线缺失"
#text="建筑物内施工垃圾的清运未采用器具或管道运输 21号楼22层西单元北外架"
text_arr=["建筑物内施工垃圾的清运未采用器具或管道运输 21号楼22层西单元北外架","用电设备未有各自专用的开关箱 用电设备未有各自开关箱箱内接地线缺失","施工人员未佩戴安全带"]



if True:
    #dataset = 'THUCNews'  # 数据集
    dataset = 'anquan'  # 数据集
    x = bert
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    #torch.cuda.manual_seed_all(1)
    #torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    config.device='cpu'

    #train_data, dev_data, test_data = build_dataset(config)
    #test_iter = build_iterator(test_data, config)

    model = x.Model(config).to(config.device)

    #model.load_state_dict(torch.load(config.save_path))
    model.load_state_dict(torch.load(config.save_path,map_location='cpu'))
    model.eval()
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
      for text in text_arr:
        texts = get_ids(text,config)
        #print(texts)
        start=time.time()
        outputs = model(texts)
        print(time.time()-start)
        predic = torch.max(outputs.data, 1)[1].cpu().numpy()
        predict_all = np.append(predict_all, predic)
    print(predict_all)




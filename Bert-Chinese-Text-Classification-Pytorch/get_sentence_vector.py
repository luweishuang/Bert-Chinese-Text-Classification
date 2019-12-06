# coding: UTF-8

import torch
import numpy as np
from importlib import import_module

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def get_sentence_vector(config, model, cur_line):
    test_data = process_data(cur_line, config, config.pad_size)
    test_iter = _to_tensor(test_data, config.device)

    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    with torch.no_grad():
        for i, (texts) in enumerate([test_iter]):
            outputs, sentence_encoders = model(texts)
            sentence_vec = sentence_encoders.data.cpu().numpy()
    return sentence_vec


def _to_tensor(datas, device):
    x = torch.LongTensor([_[0] for _ in datas]).to(device)
    # pad前的长度(超过pad_size的设为pad_size)
    seq_len = torch.LongTensor([_[1] for _ in datas]).to(device)
    mask = torch.LongTensor([_[2] for _ in datas]).to(device)
    return (x, seq_len, mask)


def process_data(cur_line, config, pad_size=32):
    contents = []
    content = cur_line.strip()
    token = config.tokenizer.tokenize(content)
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
    contents.append((token_ids, seq_len, mask))
    return contents


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    model_name = "bert"
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    # eval
    model = x.Model(config).to(config.device)

    cur_line = "名师辅导：2012考研英语虚拟语气三种用法"
    result_vec = get_sentence_vector(config, model, cur_line)
    print(result_vec.shape)
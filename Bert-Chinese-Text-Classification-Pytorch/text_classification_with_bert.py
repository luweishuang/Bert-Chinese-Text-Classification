# coding=utf-8


import json
import random
import numpy as np
import torch
from importlib import import_module
from get_sentence_vector import get_sentence_vector
PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号

from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def calc(word_vec, class_vec_mat):
    scores_list = []
    for ii in range(class_vec_mat.shape[0]):
        cur_score = cosine_similarity(word_vec.reshape(1, -1), class_vec_mat[ii, :].reshape(1, -1))
        scores_list.append(cur_score[0][0])
    index = scores_list.index(max(scores_list))
    return index


def process_valdata(config, model):
    val_file_name = "./intent_classify/val_intent.json"
    val_data_dict = json.load(open(val_file_name, "r", encoding="utf-8"))
    id2class = {}
    index = 0
    data_list_all = []
    true_label_list_all = []
    class_dict = {}
    for intent_class, intance_list in val_data_dict.items():
        choice_id = random.randint(0, len(intance_list)-1)
        class_dict[index] = intance_list.pop(choice_id)
        data_list_all += intance_list
        tmp = np.ones((len(intance_list),), dtype=int) * index
        true_label_list_all += tmp.tolist()
        assert len(data_list_all) == len(true_label_list_all)
        if index not in id2class:
            id2class[index] = intent_class
            index += 1
    print("id2class = ", id2class)
    print("class_dict = ", class_dict)

    word_vec_dim = 768
    class_vec_mat = np.zeros((len(class_dict), word_vec_dim), dtype=np.float32)
    for cur_class_id in range(len(class_dict)):
        cur_intance = class_dict[cur_class_id]
        word_vec = get_sentence_vector(config, model, cur_intance)
        class_vec_mat[cur_class_id, :] = word_vec

    predict_label_list_all = []
    for ii in range(len(data_list_all)):
        cur_line = data_list_all[ii]
        word_vec = get_sentence_vector(config, model, cur_line)
        idx = calc(word_vec, class_vec_mat)
        predict_label_list_all.append(idx)
    assert len(predict_label_list_all) == len(true_label_list_all)
    con_mat = confusion_matrix(true_label_list_all, predict_label_list_all, labels=range(len(class_dict)))    # 列方向为真实标签, 行方向为预测标签
    print(con_mat)
    accuracy = accuracy_score(true_label_list_all, predict_label_list_all)
    print(accuracy)
    f1_scores = f1_score(true_label_list_all, predict_label_list_all, average='micro')
    print(f1_scores)
    print("----process valdata done-----")


if __name__ == '__main__':
    dataset = 'intent_classify'  # 数据集

    model_name = "bert"
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    # eval
    model = x.Model(config).to(config.device)

    process_valdata(config, model)
    print("----all done-----")

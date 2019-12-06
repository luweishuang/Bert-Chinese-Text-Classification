# coding=utf-8


import json
import pickle
import numpy as np
import jieba
import pandas as pd


def process_traindata():
    train_file_name = "../data/train_intent.json"
    train_data_dict = json.load(open(train_file_name, "r", encoding="utf-8"))
    id2class = {}
    index = 0
    data_list_all = []
    true_label_list_all = []
    for intent_class, intance_list in train_data_dict.items():
        data_list_all += intance_list
        tmp = np.ones((len(intance_list),), dtype=int) * index
        true_label_list_all += tmp.tolist()
        assert len(data_list_all) == len(true_label_list_all)
        if index not in id2class:
            id2class[index] = intent_class
            index += 1
    print("id2class = ", id2class)

    original_df = pd.DataFrame({"comment": data_list_all, "sentiment": true_label_list_all})
    original_df.to_pickle("intent_classify.pkl")
    print("----process traindata done-----")


if __name__ == '__main__':
    # with open("intent_classify.pkl", 'rb') as f:
    #     all_data = pickle.load(f)
    # train = all_data.sample(int(len(all_data) * 0.9))
    # val = all_data.drop(train.index)

    process_traindata()
    print("----all done-----")

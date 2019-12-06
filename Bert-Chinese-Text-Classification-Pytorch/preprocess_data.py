# coding: UTF-8


import json
import numpy as np
import random


def write_list_2_txt(txt_file_path, cur_list):
    with open(txt_file_path, "w") as fw:
        for cur_line in cur_list:
            cur_line1 = cur_line.replace("<--->", "\t")
            fw.write(cur_line1 + "\n")


def process_traindata():
    train_file_name = "./intent_classify/train_intent.json"
    all_data_dict = json.load(open(train_file_name, "r", encoding="utf-8"))

    class2id = {}
    index = 0
    data_list_all = []
    for intent_class, intance_list in all_data_dict.items():
        for cur_instance in intance_list:
            cur_line = cur_instance + "<--->" + str(index)
            data_list_all.append(cur_line)
        if intent_class not in class2id:
            class2id[intent_class] = index
            index += 1
    print("class2id = ", class2id)
    # 对字典按value排序
    class2id_new = sorted(class2id.items(), key=lambda x: x[1], reverse=False)
    with open("./intent_classify/data/class.txt", "w") as fw:
        for key, value in class2id_new:
            fw.write(key + "\n")

    random.shuffle(data_list_all)
    split_num = [int(len(data_list_all) * 0.7), int (len(data_list_all) * 0.9)]
    train_data_list = data_list_all[:split_num[0]]
    dev_data_list = data_list_all[split_num[0]:split_num[1]]
    test_data_list = data_list_all[split_num[1]:]

    write_list_2_txt("./intent_classify/data/train.txt", train_data_list)
    write_list_2_txt("./intent_classify/data/dev.txt", dev_data_list)
    write_list_2_txt("./intent_classify/data/test.txt", test_data_list)

    print("----process traindata done-----")


if __name__ == '__main__':
    process_traindata()
    print("----all done-----")
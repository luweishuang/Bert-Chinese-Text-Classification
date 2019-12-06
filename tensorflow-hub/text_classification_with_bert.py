# coding=utf-8


import json
import random
import numpy as np
import jieba
import tensorflow as tf
import tensorflow_hub as hub

from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def get_weighted_average(We, x, w):
    """
    Compute the weighted average vectors
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in sentence i
    :param w: w[i, :] are the weights for the words in sentence i
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    n_samples = x.shape[0]
    emb = np.zeros((n_samples, We.shape[1]))
    for i in range(n_samples):
        emb[i,:] = w[i,:].dot(We[x[i,:],:]) / np.count_nonzero(w[i,:])
    return emb


def get_word_2_weight_tfidf():
    all_file_name = "../data/all_intent.json"
    all_data_dict = json.load(open(all_file_name, "r", encoding="utf-8"))
    sentences = []
    for intent_class, intance_list in all_data_dict.items():
        sentences += intance_list
    sent_words = [list(jieba.cut(sent0)) for sent0 in sentences]
    # sent_words = [list(HanLP.segment(sent0)) for sent0 in sentences]
    document = [" ".join(sent0) for sent0 in sent_words]
    tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_df=0.8, stop_words=["呀", "啊", "的", "了", "哦", "吗", "你", "我", "你们"]).fit(document)
    print("tfidf_model.vocabulary_ = ", tfidf_model.vocabulary_)

    word_vec_file_name = "../data/sgns.merge.word.json"
    ori_word_vec = json.load(open(word_vec_file_name, "r", encoding="utf-8"))
    word2id = {}
    word_vec_tot = len(ori_word_vec)
    UNK_ID = word_vec_tot
    BLANK_ID = word_vec_tot + 1
    extra_token = [UNK_ID, BLANK_ID]
    word_vec_dim = len(ori_word_vec[0]['vec'][0:])  # word向量
    word_vec_mat = np.zeros((word_vec_tot + len(extra_token), word_vec_dim), dtype=np.float32)
    for cur_id, word in enumerate(ori_word_vec):
        w = word['word']
        word2id[w] = cur_id
        word_vec_mat[cur_id, :] = word['vec']
        # embedding归一化
        word_vec_mat[cur_id] = word_vec_mat[cur_id] / np.sqrt(np.sum(word_vec_mat[cur_id] ** 2))

    df = np.zeros((len(ori_word_vec),))
    dlen = len(sentences)               # 句子的个数
    for word, num in tfidf_model.vocabulary_.items():
        if word in ori_word_vec:
            df[word2id[word]] += num
    weight4ind = {}
    for i in range(len(df)):
        weight4ind[i] = np.log2((dlen + 2.0)/(1.0 + df[i]))
    return weight4ind, ori_word_vec, word2id, word_vec_mat


def get_line_vec(cur_line):
    pass


def calc(word_vec, class_vec_mat):
    scores_list = []
    for ii in range(class_vec_mat.shape[0]):
        cur_score = cosine_similarity(word_vec.reshape(1, -1), class_vec_mat[ii, :].reshape(1, -1))
        scores_list.append(cur_score[0][0])
    index = scores_list.index(max(scores_list))
    return index


def process_valdata():
    val_file_name = "../data/val_intent.json"
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

    word_vec_dim = 512
    class_vec_mat = np.zeros((len(class_dict), word_vec_dim), dtype=np.float32)
    for cur_class_id in range(len(class_dict)):
        cur_intance = class_dict[cur_class_id]
        word_vec = get_line_vec(cur_intance)
        class_vec_mat[cur_class_id, :] = word_vec

    predict_label_list_all = []
    for ii in range(len(data_list_all)):
        cur_line = data_list_all[ii]
        word_vec, non_word_mun = get_line_vec(cur_line)
        if non_word_mun > 0:
            print(("cur_class_id: %d, class_line: %s, non_word_mun = %d") % (true_label_list_all[ii], cur_line, non_word_mun))
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
    cur_line = "不要打电话了,我要投诉"

    # Create graph and finalize (finalizing optional but recommended).
    g = tf.Graph()
    with g.as_default():
        # We will be feeding 1D tensors of text into the graph.
        text_input = tf.placeholder(dtype=tf.string, shape=[None])
        # embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
        embed = hub.Module("output/")
        embedded_text = embed(text_input)
        init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    g.finalize()

    # Create session and initialize.
    session = tf.Session(graph=g)
    session.run(init_op)

    result = session.run(embedded_text, feed_dict={text_input: [cur_line]})
    print(result)
    # process_valdata()
    print("----all done-----")

import random
import re
import datasetCapsulation as Data
from transformers import BertTokenizer
from stanfordcorenlp import StanfordCoreNLP
import torch
import probability
import all_dependancy

nlp = StanfordCoreNLP('http://localhost', port=9000)

data_path = "./data/original/v2/"
dataset_name_list = ["16res"]
dataset_type_list = ["train_triplets", "dev_triplets", "test_triplets"]

triplet_pattern = re.compile(r'[(](.*?)[)]', re.S)
aspect_and_opinion_pattern = re.compile(r'[\[](.*?)[]]', re.S)
sentiment_pattern = re.compile(r"['](.*?)[']", re.S)

forward_aspect_query_template = ["[CLS]", "what", "aspects", "?", "[SEP]"]
forward_opinion_query_template = [
    "[CLS]", "what", "opinion", "given", "the", "aspect", "?", "[SEP]"]
backward_opinion_query_template = ["[CLS]", "what", "opinions", "?", "[SEP]"]
backward_aspect_query_template = [
    "[CLS]", "what", "aspect", "does", "the", "opinion", "describe", "?", "[SEP]"]
sentiment_query_template = ["[CLS]", "what", "sentiment",
                            "given", "the", "aspect", "and", "the", "opinion", "?", "[SEP]"]

# BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentiments_mapping = {
    'POS': 0,
    'NEG': 1,
    'NEU': 2
}

positive_aspects_tag_prob, negative_aspects_tag_prob, positive_opinions_tag_prob, negative_opinions_tag_prob, aspects_tag_set, opinion_tag_set = probability.data_get_tags()

aspect_roles_set, asp_roles_prob, opinions_roles_set, opinion_roles_prob = all_dependancy.data_get_roles()


def print_QA(QA: Data.QueryAndAnswer):
    print('*' * 100)
    print(QA.line, '\n',
          ids_to_tokens(QA.forward_asp_query), '\n', QA.forward_aspect_prob, '\n', QA.forward_asp_prob, '\n',  QA.forward_asp_neg_prob, '\n',  ids_to_tokens(
              QA.forward_opi_query), '\n', QA.forward_opinion_prob, '\n',
          QA.forward_asp_query_mask, '\n', QA.forward_asp_query_seg, '\n',
          QA.forward_opi_query_mask, '\n', QA.forward_opi_query_seg, '\n',
          QA.forward_asp_answer_start, '\n', QA.forward_asp_answer_end, '\n',
          QA.forward_opi_answer_start, '\n', QA.forward_opi_answer_end, '\n',
          ids_to_tokens(QA.backward_asp_query), '\n', (QA.backward_aspect_prob), '\n', ids_to_tokens(
              QA.backward_opi_query), '\n', QA.backward_opinion_prob, '\n', QA.backward_opi_prob, '\n',
          QA.backward_opi_neg_prob, '\n',
          QA.backward_asp_query_mask, '\n', QA.backward_asp_query_seg, '\n',
          QA.backward_opi_query_mask, '\n', QA.backward_opi_query_seg, '\n',
          QA.backward_asp_answer_start, '\n', QA.backward_asp_answer_end, '\n',
          QA.backward_opi_answer_start, '\n', QA.backward_opi_answer_end, '\n',
          ids_to_tokens(QA.sentiment_query), '\n', QA.sentiment_answer, '\n',
          QA.sentiment_query_mask, '\n', QA.sentiment_query_seg, '\n')

    print(QA.line)
    token_list = ids_to_tokens(QA.forward_asp_query)
    for i in range(len(token_list)):
        if QA.forward_asp_answer_start[i] == 1:
            print('forward asp start:', token_list[i])
        if QA.forward_asp_answer_end[i] == 1:
            print('forward asp end:', token_list[i])
    token_list = ids_to_tokens(QA.forward_opi_query)
    for i in range(len(token_list)):
        for j in range(len(token_list[i])):
            if QA.forward_opi_answer_start[i][j] == 1:
                print('forward opi start:', token_list[i][j])
            if QA.forward_opi_answer_end[i][j] == 1:
                print('forward opi end:', token_list[i][j])
    token_list = ids_to_tokens(QA.backward_opi_query)
    for i in range(len(token_list)):
        if QA.backward_opi_answer_start[i] == 1:
            print('backward opi start:', token_list[i])
        if QA.backward_opi_answer_end[i] == 1:
            print('backward opi end:', token_list[i])
    token_list = ids_to_tokens(QA.backward_asp_query)
    for i in range(len(token_list)):
        for j in range(len(token_list[i])):
            if QA.backward_asp_answer_start[i][j] == 1:
                print('backward asp start:', token_list[i][j])
            if QA.backward_asp_answer_end[i][j] == 1:
                print('backward asp end:', token_list[i][j])
    token_list = ids_to_tokens(QA.sentiment_query)
    for i in range(len(token_list)):
        print('sentiment[{}]:'.format(i), QA.sentiment_answer[i])
    print('*' * 100)


def valid(QA: Data.QueryAndAnswer):

    assert len(QA.forward_asp_query) == len(QA.forward_asp_answer_start) \
        == len(QA.forward_asp_answer_end) == len(QA.forward_asp_query_mask) \
        == len(QA.forward_asp_query_seg) == len(QA.forward_asp_prob) == len(QA.forward_asp_neg_prob) == len(QA.forward_aspect_prob)
    for i in range(len(QA.forward_opi_query)):
        assert len(QA.forward_opi_query[i]) == len(QA.forward_opinion_prob[i]) == len(QA.forward_opi_answer_start[i]) \
            == len(QA.forward_opi_answer_end[i]) == len(QA.forward_opi_query_mask[i]) \
            == len(QA.forward_opi_query_seg[i])
    assert len(QA.backward_opi_query) == len(QA.backward_opi_answer_start) \
        == len(QA.backward_opi_answer_end) == len(QA.backward_opi_query_mask) \
        == len(QA.backward_opi_query_seg) == len(QA.backward_opi_prob) == len(QA.backward_opi_neg_prob) == len(QA.backward_opinion_prob)
    for i in range(len(QA.backward_asp_query)):
        assert len(QA.backward_asp_query[i]) == len(QA.backward_asp_answer_start[i]) \
            == len(QA.backward_asp_answer_end[i]) == len(QA.backward_asp_query_mask[i]) \
            == len(QA.backward_asp_query_seg[i]) == len(QA.backward_aspect_prob[i])
    assert len(QA.sentiment_query) == len(QA.sentiment_answer) == len(QA.sentiment_query_mask) \
        == len(QA.sentiment_query_seg)
    for i in range(len(QA.sentiment_query)):
        assert len(QA.sentiment_query[i]) == len(
            QA.sentiment_query_mask[i]) == len(QA.sentiment_query_seg[i])


# ids to tokens
def ids_to_tokens(input_ids_list):
    if not isinstance(input_ids_list[0], list):
        return tokenizer.convert_ids_to_tokens(input_ids_list)
    token_list = []
    for input_ids in input_ids_list:
        token_list.append(tokenizer.convert_ids_to_tokens(input_ids))
    return token_list


# tokens to ids
def tokens_to_ids(QA_list):
    max_len = 0
    for QA in QA_list:
        QA.forward_asp_query = tokenizer.convert_tokens_to_ids(
            QA.forward_asp_query)
        if len(QA.forward_asp_query) > max_len:
            max_len = len(QA.forward_asp_query)
        for i in range(len(QA.forward_opi_query)):
            QA.forward_opi_query[i] = tokenizer.convert_tokens_to_ids(
                QA.forward_opi_query[i])
            if len(QA.forward_opi_query[i]) > max_len:
                max_len = len(QA.forward_opi_query[i])
        QA.backward_opi_query = tokenizer.convert_tokens_to_ids(
            QA.backward_opi_query)
        if len(QA.backward_opi_query) > max_len:
            max_len = len(QA.backward_opi_query)
        for i in range(len(QA.backward_asp_query)):
            QA.backward_asp_query[i] = tokenizer.convert_tokens_to_ids(
                QA.backward_asp_query[i])
            if len(QA.backward_asp_query[i]) > max_len:
                max_len = len(QA.backward_asp_query[i])
        for i in range(len(QA.sentiment_query)):
            QA.sentiment_query[i] = tokenizer.convert_tokens_to_ids(
                QA.sentiment_query[i])
            if len(QA.sentiment_query[i]) > max_len:
                max_len = len(QA.sentiment_query[i])
        valid(QA)
    return QA_list, max_len


def list_to_object(dataset_object):
    line = []
    forward_asp_query = []
    forward_asp_prob = []
    forward_asp_neg_prob = []
    forward_aspect_prob = []
    forward_opi_query = []
    forward_opinion_prob = []
    forward_asp_query_mask = []
    forward_asp_query_seg = []
    forward_opi_query_mask = []
    forward_opi_query_seg = []
    forward_asp_answer_start = []
    forward_asp_answer_end = []
    forward_opi_answer_start = []
    forward_opi_answer_end = []
    backward_asp_query = []
    backward_aspect_prob = []
    backward_opi_query = []
    backward_opi_prob = []
    backward_opi_neg_prob = []
    backward_opinion_prob = []
    backward_asp_query_mask = []
    backward_asp_query_seg = []
    backward_opi_query_mask = []
    backward_opi_query_seg = []
    backward_asp_answer_start = []
    backward_asp_answer_end = []
    backward_opi_answer_start = []
    backward_opi_answer_end = []
    sentiment_query = []
    sentiment_answer = []
    sentiment_query_mask = []
    sentiment_query_seg = []
    for QA in dataset_object:
        line.append(QA.line)
        forward_asp_query.append(QA.forward_asp_query)
        forward_asp_prob.append(QA.forward_asp_prob)
        forward_asp_neg_prob.append(QA.forward_asp_neg_prob)
        forward_aspect_prob.append(QA.forward_aspect_prob)
        forward_opi_query.append(QA.forward_opi_query)
        forward_opinion_prob.append(QA.forward_opinion_prob)
        forward_asp_query_mask.append(QA.forward_asp_query_mask)
        forward_asp_query_seg.append(QA.forward_asp_query_seg)
        forward_opi_query_mask.append(QA.forward_opi_query_mask)
        forward_opi_query_seg.append(QA.forward_opi_query_seg)
        forward_asp_answer_start.append(QA.forward_asp_answer_start)
        forward_asp_answer_end.append(QA.forward_asp_answer_end)
        forward_opi_answer_start.append(QA.forward_opi_answer_start)
        forward_opi_answer_end.append(QA.forward_opi_answer_end)
        backward_asp_query.append(QA.backward_asp_query)
        backward_aspect_prob.append(QA.backward_aspect_prob)
        backward_opi_query.append(QA.backward_opi_query)
        backward_opi_prob.append(QA.backward_opi_prob)
        backward_opi_neg_prob.append(QA.backward_opi_neg_prob)
        backward_opinion_prob.append(QA.backward_opinion_prob)
        backward_asp_query_mask.append(QA.backward_asp_query_mask)
        backward_asp_query_seg.append(QA.backward_asp_query_seg)
        backward_opi_query_mask.append(QA.backward_opi_query_mask)
        backward_opi_query_seg.append(QA.backward_opi_query_seg)
        backward_asp_answer_start.append(QA.backward_asp_answer_start)
        backward_asp_answer_end.append(QA.backward_asp_answer_end)
        backward_opi_answer_start.append(QA.backward_opi_answer_start)
        backward_opi_answer_end.append(QA.backward_opi_answer_end)
        sentiment_query.append(QA.sentiment_query)
        sentiment_answer.append(QA.sentiment_answer)
        sentiment_query_mask.append(QA.sentiment_query_mask)
        sentiment_query_seg.append(QA.sentiment_query_seg)

    return Data.QueryAndAnswer(line=line,
                               forward_asp_query=forward_asp_query,
                               forward_asp_prob=forward_asp_prob,
                               forward_asp_neg_prob=forward_asp_neg_prob,
                               forward_aspect_prob=forward_aspect_prob,
                               forward_opi_query=forward_opi_query,
                               forward_opinion_prob=forward_opinion_prob,
                               forward_asp_query_mask=forward_asp_query_mask,
                               forward_asp_query_seg=forward_asp_query_seg,
                               forward_opi_query_mask=forward_opi_query_mask,
                               forward_opi_query_seg=forward_opi_query_seg,
                               forward_asp_answer_start=forward_asp_answer_start,
                               forward_asp_answer_end=forward_asp_answer_end,
                               forward_opi_answer_start=forward_opi_answer_start,
                               forward_opi_answer_end=forward_opi_answer_end,
                               backward_asp_query=backward_asp_query,
                               backward_aspect_prob=backward_aspect_prob,
                               backward_opi_query=backward_opi_query,
                               backward_opi_prob=backward_opi_prob,
                               backward_opi_neg_prob=backward_opi_neg_prob,
                               backward_opinion_prob=backward_opinion_prob,
                               backward_asp_query_mask=backward_asp_query_mask,
                               backward_asp_query_seg=backward_asp_query_seg,
                               backward_opi_query_mask=backward_opi_query_mask,
                               backward_opi_query_seg=backward_opi_query_seg,
                               backward_asp_answer_start=backward_asp_answer_start,
                               backward_asp_answer_end=backward_asp_answer_end,
                               backward_opi_answer_start=backward_opi_answer_start,
                               backward_opi_answer_end=backward_opi_answer_end,
                               sentiment_query=sentiment_query,
                               sentiment_answer=sentiment_answer,
                               sentiment_query_mask=sentiment_query_mask,
                               sentiment_query_seg=sentiment_query_seg)


def dataset_align(dataset_object, max_tokens_len, max_aspect_num):
    for dataset_type in dataset_type_list:
        tokenized_QA_list = dataset_object[dataset_type]
        for tokenized_QA in tokenized_QA_list:

            tokenized_QA.forward_asp_query.extend(
                [0] * (max_tokens_len - len(tokenized_QA.forward_asp_query)))
            tokenized_QA.forward_asp_prob.extend(
                [0] * (max_tokens_len - len(tokenized_QA.forward_asp_prob)))
            tokenized_QA.forward_asp_neg_prob.extend(
                [0] * (max_tokens_len - len(tokenized_QA.forward_asp_neg_prob)))
            tokenized_QA.forward_aspect_prob.extend(
                [0] * (max_tokens_len - len(tokenized_QA.forward_aspect_prob)))
            tokenized_QA.forward_asp_query_mask.extend(
                [0] * (max_tokens_len - len(tokenized_QA.forward_asp_query_mask)))
            tokenized_QA.forward_asp_query_seg.extend(
                [1] * (max_tokens_len - len(tokenized_QA.forward_asp_query_seg)))

            tokenized_QA.forward_asp_answer_start.extend(
                [-1] * (max_tokens_len - len(tokenized_QA.forward_asp_answer_start)))
            tokenized_QA.forward_asp_answer_end.extend(
                [-1] * (max_tokens_len - len(tokenized_QA.forward_asp_answer_end)))

            for i in range(len(tokenized_QA.forward_opi_query)):
                tokenized_QA.forward_opi_query[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.forward_opi_query[i])))
                tokenized_QA.forward_opinion_prob[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.forward_opinion_prob[i])))
                tokenized_QA.forward_opi_answer_start[i].extend(
                    [-1] * (max_tokens_len - len(tokenized_QA.forward_opi_answer_start[i])))
                tokenized_QA.forward_opi_answer_end[i].extend(
                    [-1] * (max_tokens_len - len(tokenized_QA.forward_opi_answer_end[i])))

                tokenized_QA.forward_opi_query_mask[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.forward_opi_query_mask[i])))
                tokenized_QA.forward_opi_query_seg[i].extend(
                    [1] * (max_tokens_len - len(tokenized_QA.forward_opi_query_seg[i])))

            tokenized_QA.backward_opi_query.extend(
                [0] * (max_tokens_len - len(tokenized_QA.backward_opi_query)))
            tokenized_QA.backward_opi_prob.extend(
                [0] * (max_tokens_len - len(tokenized_QA.backward_opi_prob)))
            tokenized_QA.backward_opi_neg_prob.extend(
                [0] * (max_tokens_len - len(tokenized_QA.backward_opi_neg_prob)))
            tokenized_QA.backward_opinion_prob.extend(
                [0] * (max_tokens_len - len(tokenized_QA.backward_opinion_prob)))
            tokenized_QA.backward_opi_query_mask.extend(
                [0] * (max_tokens_len - len(tokenized_QA.backward_opi_query_mask)))
            tokenized_QA.backward_opi_query_seg.extend(
                [1] * (max_tokens_len - len(tokenized_QA.backward_opi_query_seg)))

            tokenized_QA.backward_opi_answer_start.extend(
                [-1] * (max_tokens_len - len(tokenized_QA.backward_opi_answer_start)))
            tokenized_QA.backward_opi_answer_end.extend(
                [-1] * (max_tokens_len - len(tokenized_QA.backward_opi_answer_end)))

            for i in range(len(tokenized_QA.backward_asp_query)):
                tokenized_QA.backward_asp_query[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.backward_asp_query[i])))
                tokenized_QA.backward_asp_answer_start[i].extend(
                    [-1] * (max_tokens_len - len(tokenized_QA.backward_asp_answer_start[i])))
                tokenized_QA.backward_asp_answer_end[i].extend(
                    [-1] * (max_tokens_len - len(tokenized_QA.backward_asp_answer_end[i])))
                tokenized_QA.backward_aspect_prob[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.backward_aspect_prob[i])))

                tokenized_QA.backward_asp_query_mask[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.backward_asp_query_mask[i])))
                tokenized_QA.backward_asp_query_seg[i].extend(
                    [1] * (max_tokens_len - len(tokenized_QA.backward_asp_query_seg[i])))

            for i in range(len(tokenized_QA.sentiment_query)):
                tokenized_QA.sentiment_query[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.sentiment_query[i])))

                tokenized_QA.sentiment_query_mask[i].extend(
                    [0] * (max_tokens_len - len(tokenized_QA.sentiment_query_mask[i])))
                tokenized_QA.sentiment_query_seg[i].extend(
                    [1] * (max_tokens_len - len(tokenized_QA.sentiment_query_seg[i])))

            for i in range(max_aspect_num - len(tokenized_QA.forward_opi_query)):
                tokenized_QA.forward_opi_query.insert(
                    -1, tokenized_QA.forward_opi_query[0])
                tokenized_QA.forward_opi_query_mask.insert(
                    -1, tokenized_QA.forward_opi_query_mask[0])
                tokenized_QA.forward_opi_query_seg.insert(
                    -1, tokenized_QA.forward_opi_query_seg[0])

                tokenized_QA.forward_opinion_prob.insert(
                    -1, tokenized_QA.forward_opinion_prob[0])

                tokenized_QA.forward_opi_answer_start.insert(
                    -1, tokenized_QA.forward_opi_answer_start[0])
                tokenized_QA.forward_opi_answer_end.insert(
                    -1, tokenized_QA.forward_opi_answer_end[0])

                tokenized_QA.backward_asp_query.insert(
                    -1, tokenized_QA.backward_asp_query[0])
                tokenized_QA.backward_asp_query_mask.insert(
                    -1, tokenized_QA.backward_asp_query_mask[0])
                tokenized_QA.backward_asp_query_seg.insert(
                    -1, tokenized_QA.backward_asp_query_seg[0])

                tokenized_QA.backward_aspect_prob.insert(
                    -1, tokenized_QA.backward_aspect_prob[0])

                tokenized_QA.backward_asp_answer_start.insert(
                    -1, tokenized_QA.backward_asp_answer_start[0])
                tokenized_QA.backward_asp_answer_end.insert(
                    -1, tokenized_QA.backward_asp_answer_end[0])

                tokenized_QA.sentiment_query.insert(-1,
                                                    tokenized_QA.sentiment_query[0])
                tokenized_QA.sentiment_query_mask.insert(
                    -1, tokenized_QA.sentiment_query_mask[0])
                tokenized_QA.sentiment_query_seg.insert(
                    -1, tokenized_QA.sentiment_query_seg[0])

                tokenized_QA.sentiment_answer.insert(
                    -1, tokenized_QA.sentiment_answer[0])

            valid(tokenized_QA)
            if random.random() > 0.99:
                print_QA(tokenized_QA)
    return dataset_object


def get_start_end(str_list):
    index_list = [int(s) for s in str_list]
    for i in range(1, len(index_list)):
        assert index_list[i] - index_list[i - 1] == 1
    return [index_list[0], index_list[-1]]


def preprocess(sen):
    sente = " ".join(sen)
    a = re.sub('\##', '', sente)
    se = a.replace('.', 'se')
    ses = se.replace('+', 'se')
    return ses


# def get_obj_with_rel(word_list, is_aspect):
#     sen = ' '.join(word_list)
#     new_sen = word_list
#     sentenseTags = nlp.pos_tag(sen)
#     aspect_opinion_rel = ["JJ-amod-NNS", "NN-nsubj-JJR",
#                           "NN-nsubj-JJ", "NNS-nsubj-JJ", "NN-nsubj-JJS"]
#     aspects_tags = ["NN", "NNS"]
#     opinion_tags = ["JJ", "JJR", "JJS"]
#     dependency = nlp.dependency_parse(sen)
#     #print("dep", dependency)

#     aspects = []
#     opinions = []
#     for dependance in dependency:
#         role = (dependance[0]).lower()
#         cible = dependance[1]-1
#         mot = dependance[2]-1

#         full_role = str(sentenseTags[mot][1]) + "-" + \
#             role + "-" + str(sentenseTags[cible][1])

#         if full_role in aspect_opinion_rel:
#             #print(full_role)
#             tag_list = [sentenseTags[mot], sentenseTags[cible]]

#             for i in range(len(tag_list)):
#                 if tag_list[i][1] in aspects_tags:
#                     aspects.append(tag_list[i][0])
#                 else:
#                     opinions.append(tag_list[i][0])

#     #print("opinions",opinions)
#     index_list = []
#     if is_aspect == 'aspect':
#         for i in range(len(new_sen)):
#             if new_sen[i] in aspects:
#                 index_list.append(i)
#     else:
#         for i in range(len(new_sen)):
#             if new_sen[i] in opinions:
#                 index_list.append(i)
#     return index_list

#TO DO ponctuation
def get_restrictive_rel(ok_start_token, aspect_list, role_set, roles_prob):
    sent = ' '.join(ok_start_token)
    #sena = sent[:-1]
    sena = preprocess(ok_start_token)
    new_sen = sena.split(" ")

    #print(new_sen)
    sentenseTags = nlp.pos_tag(sena)
    #print(sentenseTags)

    dependency = nlp.dependency_parse(sena)
    #print("dep", dependency)
    opinions = []
    word_and_prob = []
    rep = []
    for dependance in dependency:
        word = {}
        role = (dependance[0]).lower()
        cible = dependance[1]-1
        mot = dependance[2]-1

        full_role = str(sentenseTags[mot][1]) + "-" + \
            role + "-" + str(sentenseTags[cible][1])
        word['mot'] = sentenseTags[mot][0]
        word['cible'] = sentenseTags[cible][0]
        word['role'] = full_role

        for i in range(len(aspect_list)):
            if ((sentenseTags[mot][0] == aspect_list[i])):
                if full_role in role_set:
                    opinions.append(sentenseTags[cible][0])
                    word_cible = {}
                    word_cible["mot"] = sentenseTags[mot][0]
                    word_cible["cible"] = sentenseTags[cible][0]
                    word_cible["prob"] = roles_prob[full_role]
                    word_cible["indice"] = new_sen.index(
                        sentenseTags[cible][0])
                    word_and_prob.append(word_cible)
                    #print(word_cible)
            rep.append(word)
        index_list = []
        for i in range(len(new_sen)):
            if new_sen[i] in opinions:
                index_list.append(i)
    return word_and_prob, index_list


def get_sentence_representation(pos_tag, dependancy_parse):
    sentenseTags = pos_tag
    dependency = dependancy_parse
    sentence_representation = []
    for dependance in dependency:
        word = {}
        role = (dependance[0]).lower()
        cible = dependance[1]-1
        mot = dependance[2]-1
        full_role = str(sentenseTags[mot][1]) + "-" + \
            role + "-" + str(sentenseTags[cible][1])
        word['mot'] = sentenseTags[mot][0]
        word['cible'] = sentenseTags[cible][0]
        word['full_role'] = full_role
        if full_role in aspect_roles_set:
            word['forward_opinion_prob'] = asp_roles_prob[full_role]
        if full_role in opinions_roles_set:
            word['backward_aspect_prob'] = opinion_roles_prob[full_role]
        sentence_representation.append(word)
    return sentence_representation


def make_QA(line, word_list, aspect_list, opinion_list, sentiment_list):
    #print(word_list)
    forward_asp_query = forward_aspect_query_template + word_list
    line_prop = preprocess(word_list)
    line_un_process = line_prop.split(" ")
    sentence_pos = nlp.pos_tag(line_prop)
    sentence_dependancy_parse = nlp.dependency_parse(line_prop)
    #sentence_representation = get_sentence_representation(
    #  sentence_pos, sentence_dependancy_parse)
    #assert (len(word_list)) == len(sentence_pos)
    asp_prob_template = [0] * len(forward_aspect_query_template)
    asp_prop_sen = [0] * len(word_list)

    neg_asp_template = [0] * len(forward_aspect_query_template)
    neg_asp_prop_sen = [0] * len(word_list)

    # Probabilités des aspects en fonction du POS-tagging
    for i in range(0, len(sentence_pos)-1):
        if sentence_pos[i][1] in aspects_tag_set:
            asp_prop_sen[i] = positive_aspects_tag_prob[sentence_pos[i][1]]
            neg_asp_prop_sen[i] = negative_aspects_tag_prob[sentence_pos[i][1]]
    forward_asp_prob = asp_prob_template + asp_prop_sen
    forward_asp_neg_prob = neg_asp_template + neg_asp_prop_sen

    #Probabilités des aspects en fonction des relations grammaticales
    aspect_prob_sen = [0] * len(word_list)
    for dependance in sentence_dependancy_parse:
        word = {}
        role = (dependance[0]).lower()
        cible = dependance[1]-1
        mot = dependance[2]-1
        full_role = str(sentence_pos[mot][1]) + "-" + \
            role + "-" + str(sentence_pos[cible][1])
        if full_role in aspect_roles_set:
            word["mot"] = sentence_pos[mot][0]
            word["prob"] = asp_roles_prob[full_role]
            for i in range(len(line_un_process)):
                if word["mot"] == line_un_process[i]:
                    aspect_prob_sen[i] = asp_roles_prob[full_role]

    forward_aspect_prob = asp_prob_template + aspect_prob_sen

    forward_asp_query_mask = [1] * len(forward_asp_query)
    forward_asp_query_seg = [
        0] * len(forward_aspect_query_template) + [1] * len(word_list)
    forward_asp_answer_start = [-1] * \
        len(forward_aspect_query_template) + [0] * len(word_list)
    forward_asp_answer_end = [-1] * \
        len(forward_aspect_query_template) + [0] * len(word_list)
    forward_opi_query = []
    forward_opinion_prob = []
    forward_opi_query_mask = []
    forward_opi_query_seg = []
    forward_opi_answer_start = []
    forward_opi_answer_end = []

    backward_opi_query = backward_opinion_query_template + word_list
    line_prop = preprocess(word_list)
    sentence_pos = nlp.pos_tag(line_prop)
    #assert (len(word_list)) == len(sentence_pos)

    opi_pos_prob_template = [0] * len(backward_opinion_query_template)
    opi_pos_prob_sen = [0] * len(word_list)

    neg_opi_pos_prob_template = [0] * len(backward_opinion_query_template)
    neg_opi_pos_prob_sen = [0] * len(word_list)

    #probablités des opinions en fonction de l'étiquetage morphosyntaxique
    for i in range(0, len(sentence_pos)-1):
        if sentence_pos[i][1] in opinion_tag_set:
            opi_pos_prob_sen[i] = positive_opinions_tag_prob[sentence_pos[i][1]]
            neg_opi_pos_prob_sen[i] = negative_opinions_tag_prob[sentence_pos[i][1]]

    backward_opi_prob = opi_pos_prob_template + opi_pos_prob_sen
    backward_opi_neg_prob = neg_opi_pos_prob_template + neg_opi_pos_prob_sen

    opinion_prob_sen = [0] * len(word_list)

    # Probabilités des opinions en fonction des relations grammaticales
    for dependance in sentence_dependancy_parse:
        word = {}
        role = (dependance[0]).lower()
        cible = dependance[1]-1
        mot = dependance[2]-1
        full_role = str(sentence_pos[mot][1]) + "-" + \
            role + "-" + str(sentence_pos[cible][1])
        if full_role in opinions_roles_set:
            word["mot"] = sentence_pos[mot][0]
            for i in range(len(line_un_process)):
                if word["mot"] == line_un_process[i]:
                    opinion_prob_sen[i] = opinion_roles_prob[full_role]

    backward_opinion_prob = opi_pos_prob_template + opinion_prob_sen

    backward_opi_query_mask = [1] * len(backward_opi_query)
    backward_opi_query_seg = [
        0] * len(backward_opinion_query_template) + [1] * len(word_list)
    backward_opi_answer_start = [-1] * \
        len(backward_opinion_query_template) + [0] * len(word_list)
    backward_opi_answer_end = [-1] * \
        len(backward_opinion_query_template) + [0] * len(word_list)
    backward_asp_query = []
    backward_aspect_prob = []
    backward_asp_query_mask = []
    backward_asp_query_seg = []
    backward_asp_answer_start = []
    backward_asp_answer_end = []

    sentiment_query = []
    sentiment_query_mask = []
    sentiment_query_seg = []
    sentiment_answer = sentiment_list
    sentiment_word_list = word_list[:]
    sentiment_query_mask_init = [1] * len(sentiment_word_list)
    for i in range(len(aspect_list)):
        for aspect_index in range(aspect_list[i][0], aspect_list[i][1] + 1):
            sentiment_word_list[aspect_index] = "[PAD]"
            sentiment_query_mask_init[aspect_index] = 0
        for opinion_index in range(opinion_list[i][0], opinion_list[i][1] + 1):
            sentiment_word_list[opinion_index] = "[PAD]"
            sentiment_query_mask_init[opinion_index] = 0
    assert len(forward_opinion_prob) == len(forward_opi_query)
    for i in range(len(aspect_list)):
        asp = aspect_list[i]
        opi = opinion_list[i]

        aspect = word_list[asp[0]:asp[1] + 1]

        opi_template = [0] * len(forward_opinion_query_template[0:6] + word_list[asp[0]:asp[1] + 1] +
                                 forward_opinion_query_template[6:])
        opi_sentence = [0] * len(word_list)

        opin_list, opi_indexes = get_restrictive_rel(
            word_list, aspect, aspect_roles_set, asp_roles_prob)
        #print("forward_opi",opin_list)
        if (opin_list):
            for i in range(len(opin_list)):
                opi_sentence[opin_list[i].get(
                    "indice")] = opin_list[i].get("prob")
        opi_vector = opi_template + opi_sentence
        #print("for_prob",opi_sentence)
        forward_opinion_prob.append(opi_vector)

        forward_asp_answer_start[len(
            forward_aspect_query_template) + asp[0]] = 1
        forward_asp_answer_end[len(forward_aspect_query_template) + asp[1]] = 1

        opi_query_temp = forward_opinion_query_template[0:6] + word_list[asp[0]:asp[1] + 1] + \
            forward_opinion_query_template[6:] + word_list
        forward_opi_query.append(opi_query_temp)

        opi_query_mask_temp = [1] * len(opi_query_temp)
        opi_query_seg_temp = [
            0] * (len(opi_query_temp) - len(word_list)) + [1] * len(word_list)
        forward_opi_query_mask.append(opi_query_mask_temp)
        forward_opi_query_seg.append(opi_query_seg_temp)

        opi_answer_start_temp = [-1] * \
            (len(opi_query_temp) - len(word_list)) + [0] * len(word_list)
        opi_answer_start_temp[len(opi_query_temp) -
                              len(word_list) + opi[0]] = 1
        opi_answer_end_temp = [-1] * (len(opi_query_temp) -
                                      len(word_list)) + [0] * len(word_list)
        opi_answer_end_temp[len(opi_query_temp) - len(word_list) + opi[1]] = 1
        forward_opi_answer_start.append(opi_answer_start_temp)
        forward_opi_answer_end.append(opi_answer_end_temp)

        backward_opi_answer_start[len(
            backward_opinion_query_template) + opi[0]] = 1
        backward_opi_answer_end[len(
            backward_opinion_query_template) + opi[1]] = 1

        opinion = word_list[opi[0]:opi[1] + 1]

        asp_template = [0] * len(backward_aspect_query_template[0:6] + word_list[opi[0]:opi[1] + 1] +
                                 backward_aspect_query_template[6:])
        asp_sentence = [0] * len(word_list)

        asp_list, aspect_indexes = get_restrictive_rel(
            word_list, opinion, opinions_roles_set, opinion_roles_prob)
        #print("backward_aspect", asp_list)
        if asp_list:
            for i in range(len(asp_list)):
                asp_sentence[asp_list[i].get(
                    "indice")] = asp_list[i].get("prob")
        #print("prob_vector",asp_sentence )
        asp_vector = asp_template + asp_sentence

        backward_aspect_prob.append(asp_vector)

        asp_query_temp = backward_aspect_query_template[0:6] + word_list[opi[0]:opi[1] + 1] + \
            backward_aspect_query_template[6:] + word_list
        backward_asp_query.append(asp_query_temp)

        asp_query_mask_temp = [1] * len(asp_query_temp)
        asp_query_seg_temp = [
            0] * (len(asp_query_temp) - len(word_list)) + [1] * len(word_list)
        backward_asp_query_mask.append(asp_query_mask_temp)
        backward_asp_query_seg.append(asp_query_seg_temp)

        asp_answer_start_temp = [-1] * \
            (len(asp_query_temp) - len(word_list)) + [0] * len(word_list)
        asp_answer_start_temp[len(asp_query_temp) -
                              len(word_list) + asp[0]] = 1
        asp_answer_end_temp = [-1] * (len(asp_query_temp) -
                                      len(word_list)) + [0] * len(word_list)
        asp_answer_end_temp[len(asp_query_temp) - len(word_list) + asp[1]] = 1
        backward_asp_answer_start.append(asp_answer_start_temp)
        backward_asp_answer_end.append(asp_answer_end_temp)

        sentiment_word_list_temp = sentiment_word_list[:]
        sentiment_query_mask_init_temp = sentiment_query_mask_init[:]
        for aspect_index in range(asp[0], asp[1] + 1):
            sentiment_word_list_temp[aspect_index] = word_list[aspect_index]
            sentiment_query_mask_init_temp[aspect_index] = 1
        for opinion_index in range(opi[0], opi[1] + 1):
            sentiment_word_list_temp[opinion_index] = word_list[opinion_index]
            sentiment_query_mask_init_temp[opinion_index] = 1

        sentiment_query_temp = sentiment_query_template[0:6] + word_list[asp[0]:asp[1] + 1] + \
            sentiment_query_template[6:9] + word_list[opi[0]:opi[1] + 1] + \
            sentiment_query_template[9:] + sentiment_word_list_temp
        sentiment_query.append(sentiment_query_temp)
        sentiment_query_mask_temp = [1] * (len(sentiment_query_temp) - len(sentiment_word_list_temp)) + \
            sentiment_query_mask_init_temp
        sentiment_query_seg_temp = [0] * (len(sentiment_query_temp) - len(sentiment_word_list_temp)) + \
                                   [1] * len(sentiment_word_list_temp)
        sentiment_query_mask.append(sentiment_query_mask_temp)
        sentiment_query_seg.append(sentiment_query_seg_temp)
    assert len(forward_opinion_prob) == len(forward_opi_query)

    return Data.QueryAndAnswer(line=line,
                               forward_asp_query=forward_asp_query,
                               forward_asp_prob=forward_asp_prob,
                               forward_asp_neg_prob=forward_asp_neg_prob,
                               forward_aspect_prob=forward_aspect_prob,
                               forward_opi_query=forward_opi_query,
                               forward_opinion_prob=forward_opinion_prob,
                               forward_asp_query_mask=forward_asp_query_mask,
                               forward_asp_query_seg=forward_asp_query_seg,
                               forward_opi_query_mask=forward_opi_query_mask,
                               forward_opi_query_seg=forward_opi_query_seg,
                               forward_asp_answer_start=forward_asp_answer_start,
                               forward_asp_answer_end=forward_asp_answer_end,
                               forward_opi_answer_start=forward_opi_answer_start,
                               forward_opi_answer_end=forward_opi_answer_end,
                               backward_asp_query=backward_asp_query,
                               backward_aspect_prob=backward_aspect_prob,
                               backward_opi_query=backward_opi_query,
                               backward_opi_prob=backward_opi_prob,
                               backward_opi_neg_prob=backward_opi_neg_prob,
                               backward_opinion_prob=backward_opinion_prob,
                               backward_asp_query_mask=backward_asp_query_mask,
                               backward_asp_query_seg=backward_asp_query_seg,
                               backward_opi_query_mask=backward_opi_query_mask,
                               backward_opi_query_seg=backward_opi_query_seg,
                               backward_asp_answer_start=backward_asp_answer_start,
                               backward_asp_answer_end=backward_asp_answer_end,
                               backward_opi_answer_start=backward_opi_answer_start,
                               backward_opi_answer_end=backward_opi_answer_end,
                               sentiment_query=sentiment_query,
                               sentiment_answer=sentiment_answer,
                               sentiment_query_mask=sentiment_query_mask,
                               sentiment_query_seg=sentiment_query_seg)


def encode_solve(word_list, aspect_list, opinion_list):
    end = -1
    index = []
    new_word_list = []
    new_aspect_list = []
    new_opinion_list = []
    for word in word_list:
        encode_words = tokenizer.convert_ids_to_tokens(tokenizer.encode(word))
        encode_words_len = len(encode_words)
        for i in range(1, encode_words_len - 1):
            new_word_list.append(encode_words[i])
        start = end + 1
        end = start + encode_words_len - 3
        index.append([start, end])
    for i in range(len(aspect_list)):
        new_aspect_list.append(
            [index[aspect_list[i][0]][0], index[aspect_list[i][1]][1]])
        new_opinion_list.append(
            [index[opinion_list[i][0]][0], index[opinion_list[i][1]][1]])
    return new_word_list, new_aspect_list, new_opinion_list


def line_data_process(line, isQA=True):
    # Line sample:
    # It is easy to use , fast and has great graphics for the money .####[([10], [9], 'POS'), ([4], [2], 'POS')]
    split = line.split("####")
    # print(split)
    assert len(split) == 2
    max_aspect_num = 0
    max_len = 0
    word_list = split[0].split()
    word_list = [word.lower() for word in word_list]
    triplet_str_list = re.findall(triplet_pattern, split[1])
    aspect_list = [re.findall(aspect_and_opinion_pattern, triplet)[
        0] for triplet in triplet_str_list]
    aspect_list = [get_start_end(aspect.split(',')) for aspect in aspect_list]
    if len(aspect_list) > max_aspect_num:
        max_aspect_num = len(aspect_list)
    opinion_list = [re.findall(aspect_and_opinion_pattern, triplet)[
        1] for triplet in triplet_str_list]
    opinion_list = [get_start_end(opinion.split(','))
                    for opinion in opinion_list]
    sentiment_list = [sentiments_mapping[re.findall(sentiment_pattern, triplet)[
        0]] for triplet in triplet_str_list]
    assert len(aspect_list) > 0 and len(
        opinion_list) > 0 and len(sentiment_list) > 0
    assert len(aspect_list) == len(opinion_list) == len(sentiment_list)
    # TODO
    word_list, aspect_list, opinion_list = encode_solve(
        word_list, aspect_list, opinion_list)
    for i in range(len(aspect_list)):
        if (aspect_list[i][1] - aspect_list[i][0] + 1) > max_len:
            max_len = aspect_list[i][1] - aspect_list[i][0] + 1
        if (opinion_list[i][1] - opinion_list[i][0] + 1) > max_len:
            max_len = opinion_list[i][1] - opinion_list[i][0] + 1
    if isQA:
        return make_QA(line, word_list, aspect_list, opinion_list, sentiment_list), max_aspect_num, max_len
    else:
        return line, aspect_list, opinion_list, sentiment_list


def train_data_process(text):
    QA_list = []
    max_aspect_num = 0
    max_len = 0
    for line in text:
        QA, max_aspect_temp, max_len_temp = line_data_process(line)
        if max_aspect_temp > max_aspect_num:
            max_aspect_num = max_aspect_temp
        if max_len_temp > max_len:
            max_len = max_len_temp
        QA_list.append(QA)
    return QA_list, max_aspect_num, max_len


def test_data_process(text):
    test_dataset = []
    for line in text:
        line, aspect_list_temp, opinion_list_temp, sentiment_list_temp = line_data_process(
            line, isQA=False)
        aspect_list = []
        opinion_list = []
        asp_opi_list = []
        asp_sent_list = []
        triplet_list = []
        for i in range(0, len(aspect_list_temp)):
            if aspect_list_temp[i] not in aspect_list:
                aspect_list.append(aspect_list_temp[i])
            if opinion_list_temp[i] not in opinion_list:
                opinion_list.append(opinion_list_temp[i])
            asp_opi_temp = [aspect_list_temp[i][0], aspect_list_temp[i][1], opinion_list_temp[i][0],
                            opinion_list_temp[i][1]]
            asp_sent_temp = [aspect_list_temp[i][0],
                             aspect_list_temp[i][1], sentiment_list_temp[i]]
            triplet_temp = [aspect_list_temp[i][0], aspect_list_temp[i][1], opinion_list_temp[i][0],
                            opinion_list_temp[i][1], sentiment_list_temp[i]]
            asp_opi_list.append(asp_opi_temp)
            if asp_sent_temp not in asp_sent_list:
                asp_sent_list.append(asp_sent_temp)
            triplet_list.append(triplet_temp)
        test_dataset.append(Data.TestDataset(
            line=line,
            aspect_list=aspect_list,
            opinion_list=opinion_list,
            asp_opi_list=asp_opi_list,
            asp_sent_list=asp_sent_list,
            triplet_list=triplet_list
        ))
    return test_dataset


def dataset_process():
    for dataset_name in dataset_name_list:
        train_output_path = "./data/preprocess/v2/" + dataset_name + ".pt"
        test_output_path = "./data/preprocess/v2/" + dataset_name + "_test.pt"
        train_dataset_object = {}
        test_dataset_object = {}
        max_tokens_len = 0
        max_aspect_num = 0
        max_len = 0
        for dataset_type in dataset_type_list:
            file = open(data_path + dataset_name + "/" +
                        dataset_type + ".txt", "r", encoding="utf-8")
            text_lines = file.readlines()
            QA_list, max_aspect_temp, max_len_temp = train_data_process(
                text_lines)
            train_dataset_object[dataset_type], max_tokens_temp = tokens_to_ids(
                QA_list)
            test_dataset_object[dataset_type] = test_data_process(text_lines)
            if max_tokens_temp > max_tokens_len:
                max_tokens_len = max_tokens_temp
            if max_aspect_temp > max_aspect_num:
                max_aspect_num = max_aspect_temp
            if max_len_temp > max_len:
                max_len = max_len_temp
        train_dataset_object = dataset_align(
            train_dataset_object, max_tokens_len, max_aspect_num)
        train_dataset_object['max_tokens_len'] = max_tokens_len
        train_dataset_object['max_aspect_num'] = max_aspect_num
        train_dataset_object['max_len'] = max_len
        torch.save(train_dataset_object, train_output_path)
        torch.save(test_dataset_object, test_output_path)


if __name__ == '__main__':
    dataset_process()

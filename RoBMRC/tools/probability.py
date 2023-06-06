from stanfordcorenlp import StanfordCoreNLP
from transformers import BertTokenizer
import re

nlp = StanfordCoreNLP('http://localhost', port=9000)

data_path = "./data/original/v2/"
dataset_name_list = ["14lap"]
dataset_type_list = ["train_triplets", "dev_triplets"]


triplet_pattern = re.compile(r'[(](.*?)[)]', re.S)
aspect_and_opinion_pattern = re.compile(r'[\[](.*?)[]]', re.S)


def get_list(str_list):
    index_list = [int(s) for s in str_list]
    for i in range(1, len(index_list)):
        assert index_list[i] - index_list[i - 1] == 1
    return [index_list[i] for i in range(0, len(index_list))]


# fonction qui recupere les tags d'une phrase
def get_line_tag(line):
    split = line.split("####")
    sentence = split[0]
    sen_to_list = list(sentence.split(" "))
    triplets = split[1]
    tag_list = []
    sentence_pos = nlp.pos_tag(sentence)
    for i in range(len(sentence_pos)):
        tag_list.append(sentence_pos[i][1])

    triplet_str_list = re.findall(triplet_pattern, split[1])
    aspect_list = [re.findall(aspect_and_opinion_pattern, triplet)[
        0] for triplet in triplet_str_list]
    total_list = []
    aspect_list = [get_list(aspect.split(',')) for aspect in aspect_list]
    for i in range(len(aspect_list)):
        aspect_list_words = []
        for j in range(len(aspect_list[i])):
            aspect_list_words.append(sen_to_list[aspect_list[i][j]])
        total_list.append(aspect_list_words)
    aspect_tags = []
    # print("total_asp", total_list)
    for i in range(len(total_list)):
        for j in range(len(total_list[i])):
            for k in range(len(sentence_pos)):
                if total_list[i][j] == sentence_pos[k][0]:
                    aspect_tags.append(sentence_pos[k][1])

    opinion_list = [re.findall(aspect_and_opinion_pattern, triplet)[
        1] for triplet in triplet_str_list]
    total_opinion_list = []
    opinion_list = [get_list(opinion.split(',')) for opinion in opinion_list]
    for i in range(len(opinion_list)):
        opinion_list_words = []
        for j in range(len(opinion_list[i])):
            opinion_list_words.append(sen_to_list[opinion_list[i][j]])
        total_opinion_list.append(opinion_list_words)
    opinion_tags = []
    # print("total_asp", total_list)
    for i in range(len(total_opinion_list)):
        for j in range(len(total_opinion_list[i])):
            for k in range(len(sentence_pos)):
                if total_opinion_list[i][j] == sentence_pos[k][0]:
                    opinion_tags.append(sentence_pos[k][1])

    return tag_list, aspect_tags, opinion_tags


# fonction qui récupere les tags de tout le fichier
def train_data_get_tag(text):
    pos_tag_list = []
    aspect_tag_list = []
    opinion_tag_list = []
    for line in text:
        data, asp_tag, opi_tag = get_line_tag(line)
        pos_tag_list = pos_tag_list + data
        aspect_tag_list = aspect_tag_list + asp_tag
        opinion_tag_list = opinion_tag_list + opi_tag
    return pos_tag_list, aspect_tag_list, opinion_tag_list


# Recuperation de tous les tags
def data_get_tags():

    for dataset_name in dataset_name_list:
        total_tags = []
        total_asp_tags = []
        total_opi_tags = []
        for dataset_type in dataset_type_list:
            file = open(data_path + dataset_name + "/" +
                        dataset_type + ".txt", "r", encoding="utf-8")
            text_lines = file.readlines()

            data_preprocessed, aspects_tags, opinion_tags = train_data_get_tag(
                text_lines)
            total_tags = total_tags + data_preprocessed
            total_asp_tags = total_asp_tags + aspects_tags
            total_opi_tags = total_opi_tags + opinion_tags

        tags_list, aspects_tags, opinion_tags = total_tags, total_asp_tags, total_opi_tags
        tag_set = set(tags_list)
        tags = {}
        for i in tag_set:
            tags[i] = tags_list.count(i)

        aspects_tag_set = set(aspects_tags)
        aspects_tag = {}
        for i in aspects_tag_set:
            aspects_tag[i] = aspects_tags.count(i)

        opinion_tag_set = set(opinion_tags)
        opinions_tag = {}
        for i in opinion_tag_set:
            opinions_tag[i] = opinion_tags.count(i)
        positive_aspects_tag_prob = {}
        negative_aspects_tag_prob = {}

        for key, value in aspects_tag.items():
            if key in tag_set:
                positive_aspects_tag_prob[key] = aspects_tag[key] / tags[key]
                negative_aspects_tag_prob[key] = (
                    1 - (aspects_tag[key] / tags[key]))

        positive_opinions_tag_prob = {}
        negative_opinions_tag_prob = {}
        for key, value in opinions_tag.items():
            if key in tag_set:
                positive_opinions_tag_prob[key] = opinions_tag[key] / tags[key]
                negative_opinions_tag_prob[key] = (
                    1 - (opinions_tag[key] / tags[key]))

        return positive_aspects_tag_prob, negative_aspects_tag_prob, positive_opinions_tag_prob, negative_opinions_tag_prob, aspects_tag_set, opinion_tag_set

from stanfordcorenlp import StanfordCoreNLP
from transformers import BertTokenizer
import re

nlp = StanfordCoreNLP('http://localhost', port=9000)

data_path = "./data/original/v2/"
dataset_name_list = ["14lap"]
dataset_type_list = ["train_triplets", "dev_triplets"]

aspect_opinion_rel = ["JJ-amod-NN",
                      "JJ-amod-NNS", "NN-nsubj-JJR", "NN-nsubj-JJ", "NNS-nsubj-JJ", "NN-nsubj-JJS"]
aspect_rel = ["NN-compound-NN",]

triplet_pattern = re.compile(r'[(](.*?)[)]', re.S)
aspect_and_opinion_pattern = re.compile(r'[\[](.*?)[]]', re.S)


def get_list(str_list):
    index_list = [int(s) for s in str_list]
    for i in range(1, len(index_list)):
        assert index_list[i] - index_list[i - 1] == 1
    return [index_list[i] for i in range(0, len(index_list))]


# fonction qui recupere les tags d'une phrase
def get_line_relations(line):
    sentence_represententation = []
    split = line.split("####")
    sentence = split[0]


    sen_to_list = list(sentence.split(" "))
    triplet_str_list = re.findall(triplet_pattern, split[1])
    aspect_list = [re.findall(aspect_and_opinion_pattern, triplet)[
        0] for triplet in triplet_str_list]
    total_list = []
    aspect_list = [get_list(aspect.split(','))
                for aspect in aspect_list]
    
    for i in range(len(aspect_list)):
        aspect_list_words = []
        for j in range(len(aspect_list[i])):
            aspect_list_words.append(sen_to_list[aspect_list[i][j]])
        total_list.append(aspect_list_words)
    new_total_list = [' '.join(ele) for ele in total_list]
    sentence_represententation.append(line)

    sentence_represententation.append(new_total_list)
    
    sentenseTags = nlp.pos_tag(split[0])
    print('pos', sentenseTags)
    dependency = nlp.dependency_parse(split[0])
    print("dep", dependency)
    
    for dependance in dependency:
        word = {}
        role = (dependance[0]).lower()
        cible = dependance[1]-1
        mot = dependance[2]-1
        full_role = str(sentenseTags[mot][1]) + "-" + \
            role + "-" + str(sentenseTags[cible][1])
        print(full_role)
        word['mot'] = sentenseTags[mot][0]
        word['cible'] = sentenseTags[cible][0]
        word['full_role'] = full_role
        sentence_represententation.append(word)

    return sentence_represententation


# fonction qui récupere les tags de tout le fichier
def train_data_get_rel(text):
    data_repr = []
    for line in text:
        line_rep = get_line_relations(line)
        data_repr.append(line_rep)
    return data_repr


# Recuperation de tous les tags
def data_get_tags():

    for dataset_name in dataset_name_list:
        for dataset_type in dataset_type_list:
            file = open(data_path + dataset_name + "/" +
                        dataset_type + ".txt", "r", encoding="utf-8")
            text_lines = file.readlines()
            data_preprocessed = train_data_get_rel(
                text_lines)
            with open("../relations" + "/" + dataset_type + ".txt", "w+") as f:
                f.write("\n" + str(data_preprocessed))

        return 


data_get_tags()
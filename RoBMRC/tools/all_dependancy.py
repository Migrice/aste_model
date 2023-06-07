from stanfordcorenlp import StanfordCoreNLP
import re

nlp = StanfordCoreNLP('http://localhost', port=9000)

data_path = "./data/original/v2/"
dataset_name_list = ["15res"]
dataset_type_list = ["train_triplets", "dev_triplets"]


triplet_pattern = re.compile(r'[(](.*?)[)]', re.S)
aspect_and_opinion_pattern = re.compile(r'[\[](.*?)[]]', re.S)


def get_list(str_list):
    index_list = [int(s) for s in str_list]
    for i in range(1, len(index_list)):
        assert index_list[i] - index_list[i - 1] == 1
    return [index_list[i] for i in range(0, len(index_list))]


def get_line_relations(line):
    sentence_represententation = []
    split = line.split("####")
    sentence = split[0]
    sentence = sentence[:-1]

    sen_to_list = list(sentence.split(" "))
    triplet_str_list = re.findall(triplet_pattern, split[1])
    aspect_list = [re.findall(aspect_and_opinion_pattern, triplet)[
        0] for triplet in triplet_str_list]
    aspect_list = [get_list(aspect.split(','))
                   for aspect in aspect_list]

    opinion_list = [re.findall(aspect_and_opinion_pattern, triplet)[
        1] for triplet in triplet_str_list]
    opinion_list = [get_list(opinion.split(','))
                    for opinion in opinion_list]

    my_list = []
    for i in range(len(aspect_list)):
        for j in range(len(aspect_list[i])):
            my_list.append(sen_to_list[aspect_list[i][j]])

    my_opi_list = []
    for i in range(len(opinion_list)):
        for j in range(len(opinion_list[i])):
            my_opi_list.append(sen_to_list[opinion_list[i][j]])

    sentenseTags = nlp.pos_tag(sentence)
    dependency = nlp.dependency_parse(sentence)
    all_roles = []
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
        sentence_represententation.append(word)
        all_roles.append(full_role)

    aspect_and_roles = []
    opinion_and_roles = []
    good_rel = []
    good_opinion_rel = []
    initial_asp_tag = []
    for i in sentence_represententation:
        asp_tag = {}
        opi_tag = {}
        if i.get("mot") in my_list:
            asp_tag['mot'] = i.get("mot")
            asp_tag['cible'] = i.get("cible")
            asp_tag['role'] = i.get("full_role")
            aspect_and_roles.append(asp_tag)
            initial_asp_tag.append(i.get("full_role"))
            if (i.get("cible") in my_opi_list):
                good_rel.append(i.get("full_role"))
        if i.get("mot") in my_opi_list:
            opi_tag['mot'] = i.get("mot")
            opi_tag['cible'] = i.get("cible")
            opi_tag['role'] = i.get("full_role")
            opinion_and_roles.append(opi_tag)
            if i.get("cible") in my_list:
                good_opinion_rel.append(i.get("full_role"))

    obj = {}
    obj["line"] = line
    obj["aspect"] = aspect_and_roles
    obj["opinion"] = opinion_and_roles
    return obj, good_rel, initial_asp_tag, all_roles, good_opinion_rel


def train_data_get_rel(text):
    data_repr = []
    good_relations = []
    initial_asp_roles = []
    all_roles = []
    good_opinion_roles = []
    for line in text:
        line_rep, goo_rel, init_roles, all_rol, good_opi_rol = get_line_relations(
            line)
        data_repr.append(line_rep)
        good_relations = good_relations + goo_rel
        initial_asp_roles = initial_asp_roles + init_roles
        all_roles = all_roles + all_rol
        good_opinion_roles = good_opinion_roles + good_opi_rol
    return data_repr, good_relations, initial_asp_roles, all_roles, good_opinion_roles


def data_get_roles():

    for dataset_name in dataset_name_list:
        total_good_roles = []  # bon couple aspect-opinion
        total_all_roles = []  # toutes les relations existantes
        total_good_opinion_roles = []  # bon couple opinion-aspect
        for dataset_type in dataset_type_list:
            file = open(data_path + dataset_name + "/" +
                        dataset_type + ".txt", "r", encoding="utf-8")
            text_lines = file.readlines()
            data_preprocessed, good_aspects_roles, initial_asp_roles, all_roles, good_opi_rol = train_data_get_rel(
                text_lines)
            with open("../asp_opi_rel" + "/" + dataset_type + ".txt", "w+") as f:
                f.write("\n" + str(data_preprocessed))
            total_good_roles = total_good_roles + good_aspects_roles
            total_all_roles = total_all_roles + all_roles
            total_good_opinion_roles = total_good_opinion_roles + good_opi_rol

        aspects_tag_set = set(total_good_roles)
        aspects_tag = {}
        for i in aspects_tag_set:
            aspects_tag[i] = total_good_roles.count(i)

        all_role_set = set(total_all_roles)
        all_role = {}
        for i in all_role_set:
            all_role[i] = total_all_roles.count(i)

        aspect_roles_prob = {}
        for key, value in aspects_tag.items():
            aspect_roles_prob[key] = aspects_tag[key] / all_role[key]

        opinion_tag_set = set(total_good_opinion_roles)
        opinions_tag = {}
        for i in opinion_tag_set:
            opinions_tag[i] = total_good_opinion_roles.count(i)

        opinion_rolesprob = {}
        for key, value in opinions_tag.items():
            opinion_rolesprob[key] = opinions_tag[key] / all_role[key]

        return aspects_tag_set, aspect_roles_prob, opinion_tag_set, opinion_rolesprob

from transformers import BertTokenizer
from stanfordcorenlp import StanfordCoreNLP


sent = ['[CLS]', 'what', 'aspects', '?', '[SEP]', 'the', 'food', 'is', 'very', 'average', '.', '.', '.', 'the', 'thai', 'fusion', 'stuff', 'is', 'a', 'bit', 'too', 'sweet', ',', 'every', 'thing', 'they', 'serve', 'is', 'too', 'sweet', 'here', '.', '[PAD]',
       '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']

nlp = StanfordCoreNLP('http://localhost', port=9000)

sentences = [101, 2054, 5448, 2445, 1996, 7814, 2173, 1029, 102, 13325, 2013, 3025,
             8466, 2023, 2109, 2000, 2022, 1037, 2204, 2173, 1010, 2021, 2025, 2151, 2936, 1012]


print(sentences)
sentence = [101,  2054, 10740,  1029,   102,  1996,  2833,  2003,  2200,  2779,
            1012,  1012,  1012,  1996,  7273, 10077,  4933,  2003,  1037,  2978,
            2205,  4086,  1010,  2296,  2518,  2027,  3710,  2003,  2205,  4086,
            2182,  1012,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0]
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
a = tokenizer.convert_ids_to_tokens(sentences)
print(a)
# ne = ' '.join(a)
# print(ne)
# print(nlp.pos_tag(ne))

# def preprocess(sen):
#     a = tokenizer.convert_ids_to_tokens(sen)
#     b = ' '.join(a)
#     return b


# sen = preprocess(sentence)
# print(sen)
# new_sen = tokenizer.convert_ids_to_tokens(sentence)
# print("token list",tokenizer.convert_ids_to_tokens(sentence))

# sentenseTags = nlp.pos_tag(sen)

# print(sentenseTags)

# aspect_opinion_rel = ["JJ-amod-NNS", "NN-nsubj-JJR",
#     "NN-nsubj-JJ", "NNS-nsubj-JJ", "NN-nsubj-JJS"]
# aspects_tags = ["NN", "NNS"]
# opinion_tags = ["JJ", "JJR", "JJS"]
# dependency = nlp.dependency_parse(sen)
# print("dep", dependency)
# rep = []
# word_list = {}
# aspect_opi_pairs = []
# aspects = []
# opinions = []
# for dependance in dependency:
#     word = {}
#     role = (dependance[0]).lower()
#     cible = dependance[1]-1
#     mot = dependance[2]-1
#     full_role = str(sentenseTags[mot][1]) + "-" + \
#         role + "-" + str(sentenseTags[cible][1])

#     if full_role in aspect_opinion_rel:
#         print(full_role)
#         tag_list = [sentenseTags[mot], sentenseTags[cible]]
        
#         for i in range(len(tag_list)):
#             if tag_list[i][1] in aspects_tags:
#                 aspects.append(tag_list[i][0])
#             else:
#                 opinions.append(tag_list[i][0])

# print(aspects)
# print(opinions)

# sen_rep = {}
# index_list = []
# for i in range(len(new_sen)):
#     if new_sen[i] in aspects:
#         index_list.append(i+5)

# print(index_list)        

def get_aspect_with_rel(ok_start_token, is_aspect):
    tokenized_sen = tokenizer.convert_ids_to_tokens(ok_start_token)
    sen = ' '.join(tokenized_sen)
    new_sen = tokenizer.convert_ids_to_tokens(sentence)
    sentenseTags = nlp.pos_tag(sen)
    aspect_opinion_rel = ["JJ-amod-NNS", "NN-nsubj-JJR",
                        "NN-nsubj-JJ", "NNS-nsubj-JJ", "NN-nsubj-JJS"]
    aspects_tags = ["NN", "NNS"]
    opinion_tags = ["JJ", "JJR", "JJS"]
    dependency = nlp.dependency_parse(sen)
    #print("dep", dependency)
    
    aspects = []
    opinions = []
    for dependance in dependency:
        role = (dependance[0]).lower()
        cible = dependance[1]-1
        mot = dependance[2]-1
    
        full_role = str(sentenseTags[mot][1]) + "-" + \
            role + "-" + str(sentenseTags[cible][1])

        if full_role in aspect_opinion_rel:
            print(full_role)
            tag_list = [sentenseTags[mot], sentenseTags[cible]]

            for i in range(len(tag_list)):
                if tag_list[i][1] in aspects_tags:
                    aspects.append(tag_list[i][0])
                else:
                    opinions.append(tag_list[i][0])

    #print("opinions",opinions)
    index_list = []
    if is_aspect == 'aspect':
        for i in range(len(new_sen)):
            if new_sen[i] in aspects:
                index_list.append(i+5)
    else:
        for i in range (len(new_sen)):
            if new_sen[i] in opinions :
                index_list.append(i+5)
    return index_list    
            

#get_aspect_with_rel(sentence, 'opinion')
#print(a)


def get_restrictive_rel(sentence, aspect):
    # print("sentence",sentence)
    # print("aspects", aspect)
    # untokenized = tokenizer.convert_ids_to_tokens(sentence)
    # sen = ' '.join(untokenized)
    sen = sentence


    sentenseTags = nlp.pos_tag(sen)
    aspect_opinion_rel = ["JJ-amod-NNS", "NN-nsubj-JJR",
                          "NN-nsubj-JJ", "NNS-nsubj-JJ", "NN-nsubj-JJS", "RB-advmod-JJ"]
    aspects_tags = ["NN", "NNS"]
    opinion_tags = ["JJ", "JJR", "JJS"]
    dependency = nlp.dependency_parse(sen)
    #print("dep", dependency)

    aspects = []
    opinions = []
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
        
        if ((sentenseTags[mot][0] == aspect)):
            opinions.append(sentenseTags[cible][0])
        rep.append(word)
        
    return opinions 


# ex = "I am not a vegetarian but , almost all the dishes were great ."
# rep, opi = get_restrictive_rel(ex, "great")
# print(rep)
# print(opi)


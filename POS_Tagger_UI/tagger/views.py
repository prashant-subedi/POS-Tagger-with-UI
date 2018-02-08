from django.shortcuts import render
from django.http.response import  HttpResponse
# Create your views here.
import pickle
import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(BASE_DIR,"tagger")


def home(request):
    import numpy as np
    from .import dictionary


    if request.method == "GET":
        if 'query' in  request.GET:
            input_str = request.GET["query"]
            if input_str != "":
                amb_class_pkl = open(os.path.join(BASE_DIR ,"amb_cls.pkl"), "rb")
                amb_class  = pickle.load(amb_class_pkl)

                global_label_encoder_pkl = open(os.path.join(BASE_DIR ,"global_label_encoder.pkl"), "rb")
                global_label_encoder = pickle.load(global_label_encoder_pkl)

                global_hot_encoder_pkl = open(os.path.join(BASE_DIR ,"global_hot_encoder.pkl"), "rb")
                global_hot_encoder = pickle.load(global_hot_encoder_pkl)

                statistic_pkl = open(os.path.join(BASE_DIR ,"statistic.pkl"), "rb")
                statistic = pickle.load(statistic_pkl)

                heighest_probabilty_pkl = open(os.path.join(BASE_DIR ,"heighest_probabilty.pkl"), "rb")
                heighest_probabilty = pickle.load(heighest_probabilty_pkl)

                global_clf_pkl = open(os.path.join(BASE_DIR ,"global_clf.pkl"), "rb")
                global_clf = pickle.load(global_clf_pkl)


                ##### AI CODE
                def get_labels(l, i):
                    try:
                        return l[i]
                    except IndexError:
                        return "EMT"

                def tokenizer(s):
                    words = [x.strip() for x in s.split(" ") if x != ""]
                    words_aug = []
                    for i in words:

                        if "हरू" in i:
                            words_aug.append(i.replace("हरू", ""))
                            words_aug.append("हरू")
                        else:
                            words_aug.append(i)

                    return words_aug

                words = tokenizer(input_str)
                labeled_string = list()

                for i in words:
                    try:
                        labeled_string.append(heighest_probabilty[i])
                    except KeyError:
                        labeled_string.append("UNK")

                # Now Applying the Hybrid Approach
                # Now Applying the Hybrid Approach
                for i in range(3):
                    for e, i in enumerate(words):

                        X_raw = ((
                            get_labels(labeled_string, e - 4),
                            get_labels(labeled_string, e - 3),
                            get_labels(labeled_string, e - 2),
                            get_labels(labeled_string, e - 1),
                            get_labels(labeled_string, e + 1),
                            get_labels(labeled_string, e + 2),
                            get_labels(labeled_string, e + 3),
                            get_labels(labeled_string, e + 4),

                        ))
                        X = global_label_encoder.transform(X_raw)
                        X_one_hot = global_hot_encoder.transform(X.reshape(-1, 1))
                        X_one_hot = X_one_hot.reshape(1, -1)

                        try:
                            if len(statistic[i]) == 1:
                                pass
                            else:
                                amb_class_object = amb_class["-".join(sorted(statistic[i].keys()))]
                                clf = amb_class_object.get_clf()

                                # print(X_one_hot.shape)
                                pre = clf.predict(X_one_hot)
                                labeled_string[e] = amb_class_object.get_encoder()[0].inverse_transform([np.argmax(pre)])[0]
                        except:
                            pre = global_clf.predict(X_one_hot)
                            labeled_string[e] = global_label_encoder.inverse_transform([np.argmax(pre)])[0]

                conversion = {
                    "NN": "Common Noun",
                    "NNP": "Proper Noun",
                    "PP": "Personal Pronoun",
                    "PP$": "Possessive Pronoun",
                    "PPR": "Reflexive Pronoun",
                    "DM": "Marked Demonstrative",
                    "DUM": "Unmarked Demonstrative",
                    "VBF": "Finite Verb",
                    "VBX": "Auxillary Verb",
                    "VBI": "Verb Infinite",
                    "VBNE": "Prospective Particle",
                    "VBKO": "Aspectual particle verb",
                    "VBO": "Other particle verb",
                    "JJ": "Normal Unmarked",
                    "JJM": "Marked Adjectve",
                    "JJD": "Degree Adjectve",
                    "RBM": "Manner Adverb",
                    "RBO": "Other Adverb",
                    "INTF": "Intensifier",
                    "PLE": "Le-Postposition",
                    "PLAI": "Lai-Postposition",
                    "PKO": "Ko-Postposition",
                    "POP": "Other Postpositions",
                    "CC": "Coordinating Conjunction",
                    "CS": "Subordinating Conjunction",
                    "UH": "Interjection",
                    "CD": "Cardinal Number",
                    "OD": "Ordinal Number",
                    "HRU": "Plural marker haru",
                    "QW": "Question word",
                    "CL": "Classifier",
                    "RP": "Particle",
                    "DT": "Determiner",
                    "UNW": "Unknown word",
                    "FW": "Foreign word",
                    "YF": "sentence Final",
                    "YM": "sentence Medieval",
                    "YQ": "Quotation",
                    "YB": "Brackets",
                    "FB": "Abbreviation",
                    "ALPH": "Header List",
                    "SYM": "Symbol",
                    "<NULL>": "Null",
                }

                proper_label = [conversion[i] for  i in labeled_string]
                context = {
                "tokens": zip(words,proper_label),
                "return":True
                }
            return render(request, "ui/backgrnd.html",context=context)

    return  render(request,"ui/backgrnd.html",context={"return":False})

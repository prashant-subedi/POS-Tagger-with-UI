# Train a statistical tagger.
# Mark Amiguous data
# Make a decision tree of each amabigious class
from features import data_or_empty, set_encoder, encode_features,extract_feature
import  analytics,corpus
from  corpus import load_corpus
import  numpy as np
import sys
from dictionary import  conversion
from ambigious_class import *
a = {

    "NN":"Common Noun",
    "NNP":"Proper Noun",
}
try:
    TEST = int(sys.argv[1])
except ValueError:
    print("Enter 0,1,2 to specify testing document")
    exit()
except IndexError:
    TEST = 2

    #def get_
from sklearn.tree import DecisionTreeClassifier
train = corpus.load_corpus(all=True)
statistic = analytics.load_analytics(train)
heighest_probabilty = {}

for i in statistic:
    heighest_probabilty[i] = max(statistic[i].items(),key=lambda x:x[1])[0]

X_train_raw, Y_train_raw = extract_feature(data=train)

#Global label_encoder to encode X values
global_label_encoder,global_hot_encoder = set_encoder(Y_train_raw)
print("Training Global Classifer ....")
X_train,Y_train = encode_features(X_train_raw,Y_train_raw,global_label_encoder,global_hot_encoder)
global_clf = DecisionTreeClassifier()
global_clf.fit(X_train,Y_train)
print("Completed")

# print(train)
# Identify the ambiguity classes
amb_class = {}
for i in train:
    for x,y in enumerate(i):
        #If the word only has one tagging, we don't need a classifier
        if len(statistic[y[0]]) == 1:
            pass
        #If there is an ambiguity, we need a decission tree classifier

        else:
            cls = sorted(statistic[y[0]])
            cls_string = "-".join(cls)
            if cls_string not in amb_class:
                amb_class[cls_string] = AmbigiousClass(cls_string)
            amb_class[cls_string].add_XY(
                (data_or_empty(i, x - 4),
                data_or_empty(i,x - 3),
                data_or_empty(i,x - 2),
                data_or_empty(i, x - 1),
                data_or_empty(i, x + 1),
                data_or_empty(i, x + 2),
                data_or_empty(i, x  + 3),
                data_or_empty(i, x + 4)),
            y[1]
                )
            amb_class[cls_string].add_word(y[0])


amb_classifier = {}

print("Trainning Ambigious Class Classifiers")
for i,j in amb_class.items():
    X_raw ,Y_raw = j.get_XY()

#    print(i,len(X_raw),len(Y_raw), j.get_word())
#
#    print("*************************************")
    Z = []
    label_encoder,hot_encoder = set_encoder(Y_raw)
    j.set_encoders(label_encoder,hot_encoder)

    #Encoding X and Y using different encodings
    Y = label_encoder.transform(Y_raw)
    Y = hot_encoder.transform(Y.reshape(-1, 1))

    # This is computationally expensive task
    X = np.array([global_label_encoder.transform(i) for i in X_raw])
    Z = np.array(global_hot_encoder.transform(X[:, 0].reshape(-1, 1)))

    for i in range(1, len(X.T)):
        Z = np.append(Z, np.array(global_hot_encoder.transform(X[:, i].reshape(-1, 1))), axis=1)

    clf = DecisionTreeClassifier()
    clf.fit(Z,Y)
    j.set_clf(clf)

#print(Z.shape)
print("Completed")
def get_labels(l,i):
    try:
        return  l[i]
    except IndexError:
        return "EMT"

#Here we will classify the text:
def classify_tokenized_sentence(words,itr = 0):
    ## Initial Labeling Using Probablistic Tagger
    labeled_string = list()
    for i in words:
        try:
            labeled_string.append(heighest_probabilty[i])
        except KeyError:
            labeled_string.append("UNK")

    #Now Applying the Hybrid Approach
    # Now Applying the Hybrid Approach
    for i in range(itr):
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

    return labeled_string
import pickle

amb_class_pkl = open("amb_cls.pkl","wb")
pickle.dump(amb_class,amb_class_pkl)

global_label_encoder_pkl = open("global_label_encoder.pkl","wb")
pickle.dump(global_label_encoder,global_label_encoder_pkl)

global_hot_encoder_pkl = open("global_hot_encoder.pkl","wb")
pickle.dump(global_hot_encoder,global_hot_encoder_pkl)

statistic_pkl = open("statistic.pkl","wb")
pickle.dump(statistic,statistic_pkl)

heighest_probabilty_pkl = open("heighest_probabilty.pkl","wb")
pickle.dump(heighest_probabilty,heighest_probabilty_pkl)


global_clf_pkl = open("global_clf.pkl","wb")
pickle.dump(global_clf,global_clf_pkl)

def tokenizer(s):
    words = [x.strip() for x in s.split(" ") if x != ""]
    words_aug = []
    for i in words:

        if "हरू" in i:
            words_aug.append(i.replace("हरू",""))
            words_aug.append("हरू")
        else:
            words_aug.append(i)

    return  words_aug
while(True):
    inp =  input("Classifier_Trainned ..Enter a sentence to be classified or type 'exit' to exit \n")
    if inp == "exit":
        exit()
    else:
        words = tokenizer(inp)
        tags_predicted = classify_tokenized_sentence(words, itr=2)
        for i in range(len(words)):
            print(words[i] , conversion[tags_predicted[i]])



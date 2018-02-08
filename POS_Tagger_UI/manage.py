#!/usr/bin/env python
import os
import sys
class AmbigiousClass:
    def __init__(self,name):
        self.name = name
        self.X = []
        self.Y = []
        self.word_list =  set()

    def set_encoders(self,le,oh):
        self.label_encoder = le
        self.onehot_encoder = oh

    def get_encoder(self):
        #This encoder is only for Y values, encode X using a global encoder
        return self.label_encoder, self.onehot_encoder

    def add_XY(self,X,Y):
        self.X.append(X)
        self.Y.append(Y)

    def get_XY(self):
        return self.X,self.Y

    def add_word(self,word):
        self.word_list.add(word)

    def get_word(self):
        return self.word_list

    def set_clf(self,clf):
        self.clf = clf

    def get_clf(self):
        return self.clf

    def __str__(self):
        return "".join(self.word_list)

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "POS_Tagger_UI.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)

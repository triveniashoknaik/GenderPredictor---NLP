#GenderPredictor
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:27:15 2017

@author: triveninaik
"""

import nltk, random

class GenderApp(object):
    def __init__(self):
        names_sample = nltk.corpus.names
        self.names = [(name.lower(), 'male') for name in names_sample.words('male.txt')] + [(name.lower(), 'female') for name in names_sample.words('female.txt')]
        random.shuffle(self.names)
        self.feature_list =[(GenderApp.gender_features_part2(name), gender) for name, gender in self.names]
        self.training_dataset = self.feature_list[:4000]
        self.testing_dataset = self.feature_list[4000:]
        self.classifier = nltk.NaiveBayesClassifier.train(self.training_dataset)

    def check_gender(self,name):
        print('Name:', name, 'Gender is: ', self.classifier.classify(GenderApp.gender_features_part2(name)))

    def checking_accuracy(self):
        print('accuracy :', nltk.classify.accuracy(self.classifier, self.training_dataset))

    @staticmethod
    def gender_features_part2(word):
        word = str(word).lower()
        features = dict()
        features['first_letter'] = word[0]
        features['last_letter'] = word[-1]
        for char in 'abcdefghijklmnopqrstuvwxyz' :
            features['count' + char] = word.count(char)
            features['has' + char] = char in word
        return features

    def most_informative_features(self , n=10):
        self.classifier.show_most_informative_features(n)

if __name__ == '__main__' :
        app = GenderApp()
        app.check_gender('triveni')
        app.checking_accuracy()
        app.most_informative_features()

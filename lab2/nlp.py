from nltk.corpus import wordnet as wn
for sun in wn.synsets('CAR'):
    print(sun.name(), sun.definition())
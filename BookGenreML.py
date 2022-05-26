# chosen_genres.py =================================================================================
chosen_genres = ["Technology", "Christianity", 
"Humor", "Christmas", "Philosophy",
"Slavery", "Judaism", "Politics"]

catch_alls = ["Fiction", "War"]


# preprocessing.py =================================================================================
from gutenbergdammit.ziputils import retrieve_one
from gutenbergdammit.ziputils import searchandretrieve
from gutenbergdammit.ziputils import loadmetadata
from chosen_genres import *
import pandas as pd
import numpy as np

engl_books = []
df = pd.read_csv('./pg_catalog.csv')

df = df.dropna(subset=['Bookshelves'])
metadata = loadmetadata("gutenberg-dammit-files-v002.zip")
def convert_string_to_list(x):
    return x.split('; ')
df['Bookshelves'] = df['Bookshelves'].apply(convert_string_to_list)
all_possible_genres = {}
for genre_list in df['Bookshelves']:
    for g in genre_list:
        if g in all_possible_genres:
            all_possible_genres[g] +=1
        else:
            all_possible_genres[g] = 1
df = df.set_index(["Text#"])

def cust_func(x):
    for entry in x:
        if entry in chosen_genres:
            return entry
        else:
            for catch in catch_alls:
                if catch in entry and entry != "Historical Fiction":
                    return catch 
    return np.NaN
df['Bookshelves'] = df['Bookshelves'].apply(cust_func)

df = df.dropna(subset=['Bookshelves'])


text_nums = df.index
for data in metadata:
    if int(data["gd-num-padded"]) in text_nums and "English" in data["Language"] and (data["charset"] == "utf-8" or data["charset"] == "us-ascii"):
        engl_books.append(data["gd-num-padded"])

print(len(engl_books))
trunc = engl_books
for id in trunc:
    directory = id[0:3]
    text = retrieve_one("gutenberg-dammit-files-v002.zip", f"{directory}/{id}.txt")
    if len(text) > 1000:
        with open(f"chosen_books/{id}_trunc.txt", "w", encoding="utf-8") as f:
            f.write(text[len(text)//4:-len(text)//4])


# IPYNB File =======================================================================================

# nltk is a go to for all things nlp
import nltk
# wordnet is used for similarities between words in their definitions ex cat and dog more similar rather than cat and ball
nltk.download('wordnet')
from lexicalrichness import LexicalRichness
from nltk.corpus import wordnet as wn
from nltk.metrics.scores import precision, recall, f_measure, accuracy
import pandas as pd
import numpy as np
import os
import math
from os.path import isfile, basename, join
import re
import collections
import nltk.corpus
from chosen_genres import chosen_genres, catch_alls


def tokDoc(text):
  goodParas = []
  for para in text.split('\n'):
    goodParas.append(para)
  return goodParas


def generate_book_df():
    bookFiles = [f for f in os.listdir('./chosen_books') if isfile(join('./chosen_books', f))]
    books = []

    def clean_newlines(x):
        clean = []
        for e in x:
            if e != '\n':
                clean.append(e.strip())
        return clean

    for file in bookFiles:
        with open('./chosen_books/'+file, 'r', encoding="ISO-8859-1") as f:
            bookName = basename(file)
            bookID = int(bookName[:5])
            books.append({'bookId': bookID, 'text': clean_newlines(f.readlines())})
            # books.append({'bookId': bookID, 'text': f.read()})
    booksDf = pd.DataFrame(books)

    metadataDf = pd.read_csv('./pg_catalog.csv')
    mergedDf = pd.merge(booksDf, metadataDf, left_on='bookId', right_on='Text#')
    mergedDF = mergedDf[['bookId', 'Bookshelves', 'text']]

    # get rid of empty texts 
    mergedDF = mergedDF[mergedDF['text'].map(len) >= 1]


    def filter_genres(x):
        x_split = x.split('; ')
        for entry in x_split:
            if entry in chosen_genres:
                return entry
            else:
                for catch in catch_alls:
                    if catch in entry:
                        return catch 
        return np.NaN
    mergedDF['Bookshelves'] = mergedDF['Bookshelves'].apply(filter_genres)
    mergedDF.reset_index()
    mergedDF.columns = ['bookID', 'genre', 'text']

    def tok_sent(text):
        str_text = ' '.join(text)
        return str_text.split('.')
        

    mergedDF['sents'] = mergedDF['text'].apply(tok_sent)
    
    return mergedDF

full_book_df = generate_book_df()
full_book_df


# these two are the wordnet functions
def recGetRelated(sub, depth):
    """
    Helper 
    Recursively gets the related words up to given depth
    """
    if depth == 0:
        return set({})
    
    concepts = [sub]
    concepts += sub.hyponyms()
    
    words = set({})
    for c in concepts:
        words |= recGetRelated(c, depth - 1) # recursive call
        for l in c.lemmas():
            name = l.name()
            if '_' not in name:
                # removes multi word lemmas
                words.add(name)
    
    return words

def getRelatedWords(subject, depth):
    """
    Returns a list of words related to the given subject. Uses wordnet and explores
    hyponyms up to given depth. 
    """
    subject = subject.lower()
    words = set({})
    for c in wn.synsets(subject, pos=wn.NOUN):
        words |= recGetRelated(c, depth)
 
    return list(set(words))

genre_word_dict = {'Fiction': getRelatedWords('Fiction', 1),
 'Technology': getRelatedWords('Technology', 1),
 'Christianity': getRelatedWords('Christianity', 1),
 'Humor': getRelatedWords('Humor', 1),
 'Christmas': getRelatedWords('Christmas', 6),
 'Philosophy': getRelatedWords('Philosophy', 1),
 'Slavery': getRelatedWords('Slavery', 4),
 'Judaism': getRelatedWords('Judaism', 9),
 'Politics': getRelatedWords('Politics', 19),
 'War': getRelatedWords('War', 4)}


for e in genre_word_dict.keys():
    genre_word_dict[e] = list(set([w.lower() for w in genre_word_dict[e]]))
    
genre_word_dict


def crossValTest(training_set):
  num_folds = 10
  subset = int(len(training_set)/num_folds)
  nbAccuracy = []
  nbPrecision = []
  nbRecall = []
  nbF1 = []
  dtAccuracy = []
  dtPrecision = []
  dtRecall = []
  dtF1 = []
  for i in range(num_folds):
      test_fold = training_set.loc[i*subset:(i+1)*subset]
      train_fold = pd.concat([training_set.loc[:i*subset], training_set.loc[(i+1)*subset:]])
      featureset_train = list(zip(train_fold['features'],train_fold['genre']))
      featureset_test = list(zip(test_fold['features'],test_fold['genre']))
      naiveBayes = nltk.classify.NaiveBayesClassifier.train(featureset_train)
      decisionTree = nltk.classify.DecisionTreeClassifier.train(featureset_train)

      expected = collections.defaultdict(set)
      nbPred = collections.defaultdict(set)
      dtPred = collections.defaultdict(set)

      for j, (feats, label) in enumerate(featureset_test):
        expected[label].add(j)
        observed = naiveBayes.classify(feats)
        nbPred[observed].add(j)
        observed2 = decisionTree.classify(feats)
        dtPred[observed2].add(j)
      
      nbAccuracy.append(nltk.classify.accuracy(naiveBayes, featureset_test))
      dtAccuracy.append(nltk.classify.accuracy(decisionTree, featureset_test))
      for k in expected:
        nbPrecision.append(precision(expected[k], nbPred[k]))
        nbRecall.append(recall(expected[k], nbPred[k]))
        nbF1.append(f_measure(expected[k], nbPred[k]))

        dtPrecision.append(precision(expected[k], dtPred[k]))
        dtRecall.append(recall(expected[k], dtPred[k]))
        dtF1.append(f_measure(expected[k], dtPred[k]))

  nbAccuracy = [x for x in nbAccuracy if x]
  nbPrecision = [x for x in nbPrecision if x]
  nbRecall = [x for x in nbRecall if x]
  nbF1 = [x for x in nbF1 if x]
  dtAccuracy = [x for x in dtAccuracy if x]
  dtPrecision = [x for x in dtPrecision if x]
  dtRecall = [x for x in dtRecall if x]
  dtF1 = [x for x in dtF1 if x]

  print("CROSS VAL METRICS:")
  print("NB Accuracy:", sum(nbAccuracy)/len(nbAccuracy))
  print("NB Precision:", sum(nbPrecision)/len(nbPrecision))
  print("NB Recall:", sum(nbRecall)/len(nbRecall))
  print("NB F1:", sum(nbF1)/len(nbF1))
  print("DT Accuracy:", sum(dtAccuracy)/len(dtAccuracy))
  print("DT Precision:", sum(dtPrecision)/len(dtPrecision))
  print("DT Recall:", sum(dtRecall)/len(dtRecall))
  print("DT F1:", sum(dtF1)/len(dtF1))


  # actual metrics
def actualMetrics(training_set, test_set):
    nbPrecision = []
    nbRecall = []
    nbF1 = []
    dtPrecision = []
    dtRecall = []
    dtF1 = []

    featureset_train = list(zip(training_set['features'],training_set['genre']))
    featureset_test = list(zip(test_set['features'],test_set['genre']))
    naiveBayes = nltk.classify.NaiveBayesClassifier.train(featureset_train)
    decisionTree = nltk.classify.DecisionTreeClassifier.train(featureset_train)

    expected = collections.defaultdict(set)
    nbPred = collections.defaultdict(set)
    dtPred = collections.defaultdict(set)

    for j, (feats, label) in enumerate(featureset_test):
      expected[label].add(j)
      observed = naiveBayes.classify(feats)
      nbPred[observed].add(j)
      observed2 = decisionTree.classify(feats)
      dtPred[observed2].add(j)
      
      nbAccuracy = nltk.classify.accuracy(naiveBayes, featureset_test)
      dtAccuracy = nltk.classify.accuracy(decisionTree, featureset_test)
      for k in expected:
        nbPrecision.append(precision(expected[k], nbPred[k]))
        nbRecall.append(recall(expected[k], nbPred[k]))
        nbF1.append(f_measure(expected[k], nbPred[k]))

        dtPrecision.append(precision(expected[k], dtPred[k]))
        dtRecall.append(recall(expected[k], dtPred[k]))
        dtF1.append(f_measure(expected[k], dtPred[k]))

    nbPrecision = [x for x in nbPrecision if x]
    nbRecall = [x for x in nbRecall if x]
    nbF1 = [x for x in nbF1 if x]
    dtPrecision = [x for x in dtPrecision if x]
    dtRecall = [x for x in dtRecall if x]
    dtF1 = [x for x in dtF1 if x]

    print("PREDICTIONS:")
    print(nbPred)
    print("Expected")
    print(expected)
    print()
    print("ACTUAL METRICS:")
    print("NB Accuracy:", nbAccuracy)
    print("NB Precision:", sum(nbPrecision)/len(nbPrecision))
    print("NB Recall:", sum(nbRecall)/len(nbRecall))
    print("NB F1:", sum(nbF1)/len(nbF1))
    print()
    print("DT Accuracy:", dtAccuracy)
    print("DT Precision:", sum(dtPrecision)/len(dtPrecision))
    print("DT Recall:", sum(dtRecall)/len(dtRecall))
    print("DT F1:", sum(dtF1)/len(dtF1))


def extractFeatures(text):
  features = {}

  # Wordnet words related to person or vehicle

  lenDoc = len((' '.join(text)).split(' '))
  raw_text = ' '.join(text)
  words_text = nltk.word_tokenize(raw_text)

  def getCountI(x):
      textString = ','.join(x)
      countI = textString.count('I')
      return countI

  def getNumSentences(x):
      # have to change logic to find average length of each sentence rather than the number of sentences
      textString = ','.join(x)
      numSentences = len(re.split("[!.?]]+", textString)[0])
      return numSentences
  def getNumSentences(x):
      textString = ','.join(x)
      numSentences = len(textString.split('.'))
      return numSentences

  def getCountPeriod(x):
      textString = ','.join(x)
      countPeriod = textString.count('.')
      return countPeriod

  def getCountExclamation(x):
      textString = ','.join(x)
      countExclamation = textString.count('!')
      return countExclamation

  def getCountQuestion(x):
      textString = ','.join(x)
      countQuestion = textString.count('?')
      return countQuestion
  num_of_categories = len(genre_word_dict.keys()) # account for zero indexing

  relatedThresh = 0.001
  for genre in genre_word_dict.keys():
    count =0 
    for w in words_text:
      if w.lower() in genre_word_dict[genre]:
        if genre == 'Judaism' or genre =='Slavery':
          count +=149
        count += 1

    # features[str(genre)+'RelatedWC'] = 1 if count/lenDoc > relatedThresh else 0
    features[str(genre)+'RelatedWC'] = round(((count/lenDoc) * 100)%num_of_categories)
  
  # print(' '.join(text))
  lex = LexicalRichness(' '.join(text))
  features['lexicalrichness'] = round((lex.Herdan * 100)%num_of_categories)

# // to bin it into the num of categories
  features['countWordI'] = round(((getCountI(text)/lenDoc) * 100)%num_of_categories)
  features['numSentences'] = round(((getNumSentences(text)/lenDoc) * 100)%num_of_categories)
  features['numPeriod'] = round(((getCountPeriod(text)/lenDoc) * 100)%num_of_categories)
  features['numExclamation'] = round(((getCountExclamation(text)/lenDoc) * 100)%num_of_categories)
  features['numQuestion'] = round(((getCountQuestion(text)/lenDoc) * 100)%num_of_categories)

  return features


# actually running everything
from sklearn.model_selection import train_test_split

training_set , test_set = train_test_split(full_book_df, test_size=0.3)
training_set['features'] = training_set.apply(lambda x: extractFeatures(x['text'] ), axis=1)
test_set['features'] = test_set.apply(lambda x: extractFeatures(x['text']), axis=1)

# development purposes only
# crossValTest(training_set)
print()
actualMetrics(training_set, test_set)

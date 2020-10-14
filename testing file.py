import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Tdavidson Dataset
dataset = pd.read_csv('tdavidson_dataset.csv')


#%%
# Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
corpus2 = []
for i in range(0, 24783):
    review = re.sub('[^a-zA-Z]', ' ', dataset['tweet'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


#%%Creating the Bag of Words model

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 5:6].values

#%% Creating Tf-Idf Vextorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tfIdfVectorizer=TfidfVectorizer(use_idf=True, max_features=2000,lowercase=True, stop_words=stopwords.words('english'))
X = tfIdfVectorizer.fit_transform(corpus).toarray()
y = dataset.iloc[:, 5:6].values

#%%
tfIdfVectorizer1=TfidfVectorizer(use_idf=True, max_features=200,lowercase=True, stop_words=stopwords.words('english'))
X_test = tfIdfVectorizer1.fit_transform(corpus2).toarray()

tfidf1 = dict(zip(tfIdfVectorizer1.get_feature_names(),tfIdfVectorizer1.idf_))

#%% Tokenization
from nltk.tokenize import word_tokenize
newvec1 = [word_tokenize(titles) for titles in corpus]

#%% Word2Vec Model
from gensim.models import Word2Vec
model = Word2Vec(newvec1, sg = 1,size = 100)
result = model[model.wv.vocab]


#%% PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
r = pca.fit_transform(result)


#%% Summing up of vectors

def sum_of_vectors(model, vec):
    f1 = []
    for sent in vec:
        forged = np.zeros(100, dtype = "float32")
        count = 0
        for i in sent:
            try:
                count+=1
                forged += model[i]
            except KeyError:
                count-=1
                continue
        f1.append(forged)
    return f1

#%%
X = sum_of_vectors(model, newvec1)
y = dataset.iloc[:, 5:6].values
X = np.array(X)

#%% Doc2vec Model
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
tagged_data1 = [TaggedDocument(_d, [i]) for i, _d in enumerate(newvec1)]
model = Doc2Vec(tagged_data1, vector_size=20, window=3, epochs=25)

#%%
y = dataset.iloc[:, 5:6].values
X_rn = []
X_st = []
y_train = []
y_test = []
for i in range(24783):
    if i<18587:
        X_rn.append(tagged_data1[i])
    else:
        X_st.append(tagged_data1[i])

for i in range(24783):
    if i<18587:
        y_train.append(y[i])
    else:
        y_test.append(y[i])


#%%

def vector_for_learning(model, input_docs):
    sents = input_docs
    feature_vectors = [model.infer_vector(doc.words, steps=20) for doc in sents]
    return feature_vectors

#%%

X_train = vector_for_learning(model, X_rn)
X_test = vector_for_learning(model, X_st)



#%% Splitting the dataset into the Training set and Test set
y = dataset.iloc[:, 5:6].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(padded_corpus, y, test_size = 0.2, random_state = 0)


#%% SMOTE Oversampling
y = dataset.iloc[:, 5:6].values

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 5)
X_train_new, y_train_new = sm.fit_sample(X_train, y_train)
y_train_new = y_train_new.reshape((46014,1))

#%%

Y_train = []
Y_test = []
for i in range(len(y_train_new)):
    
    if y_train_new[i][0]==0:
        a = [1,0,0]
    elif y_train_new[i][0]==1:
        a = [0,1,0]
    else:
        a = [0,0,1]
    Y_train.append(a)
y_train_new = np.array(Y_train)

for i in range(len(y_test)):
    a = []
    if y_test[i][0]==0:
        a = [1,0,0]
    elif y_test[i][0]==1:
        a = [0,1,0]
    else:
        a = [0,0,1]
    Y_test.append(a)
y_test = np.array(Y_test)


#%% Simple Machine learning Model Training
#Fitting Methods to the Training set

from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)


#%% Deep learning Model Training
 
#Keras LSTM Model
import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from keras.layers import LSTM
from keras.layers import Dropout, Embedding

#%% Preprocessing

max_pad = 0
s = set()
for i in range(24783):
    if max_pad<len(newvec1[i]):
        max_pad = len(newvec1[i])

vocab_size = 10000
encoded_corpus = [one_hot(d,vocab_size) for d in corpus]

padded_corpus = pad_sequences(encoded_corpus, maxlen = 32, padding = 'post')


#%%

classifier = Sequential()

classifier.add(Embedding(10000,32,input_length = 32))
classifier.add(Dropout(0.2))

classifier.add(Conv1D(32,2,padding='same',activation='relu'))
classifier.add(MaxPooling1D(pool_size=2))

classifier.add(Conv1D(32,2,padding='same',activation='relu'))
classifier.add(MaxPooling1D(pool_size=2))

classifier.add(Conv1D(32,2,padding='same',activation='relu'))
classifier.add(MaxPooling1D(pool_size=2))

classifier.add(Conv1D(32,2,padding='same',activation='relu'))
classifier.add(MaxPooling1D(pool_size=2))
classifier.add(Flatten())

classifier.add(Dense(units = 3, activation = "softmax", kernel_initializer="uniform"))


classifier.compile(optimizer = "adamax",loss = "binary_crossentropy")

classifier.summary()

#%% Classifier 

classifier = Sequential()

classifier.add(Embedding(10000,32,input_length = 32))

classifier.add(LSTM(units = 32, return_sequences = True, input_shape = (200,1)))
classifier.add(Dropout(0.2))

classifier.add(LSTM(units = 32))
classifier.add(Dropout(0.2))

classifier.add(Dense(units = 3, activation = "softmax", kernel_initializer="uniform"))


classifier.compile(optimizer = "adam",loss = "categorical_crossentropy")
classifier.summary()
#%%
classifier.fit(X_train_new, y_train_new, batch_size = 32, epochs = 50, 
               validation_data = (X_test,y_test))

#%%
# Predicting the Test set results
y_pred = classifier.predict(X_test)
Y_pred = []
for i in range(len(y_pred)):
    if y_pred[i][0]>y_pred[i][1] and y_pred[i][0]>y_pred[i][2]:
        a = [1,0,0]
    if y_pred[i][1]>y_pred[i][0] and y_pred[i][1]>y_pred[i][2]:
        a = [0,1,0]
    if y_pred[i][2]>y_pred[i][1] and y_pred[i][2]>y_pred[i][0]:
        a = [0,0,1]
    Y_pred.append(a)

#%%
pred = []
for i in range(len(Y_pred)):
    if Y_pred[i][0]==1:
        pred.append([0])
    if Y_pred[i][1]==1:
        pred.append([1])
    if Y_pred[i][2]==1:
        pred.append([2])

pred = np.array(pred)

#%%
Y_test = []
for i in range(len(y_test)):
    if y_test[i][0]==1:
        Y_test.append([0])
    if y_test[i][1]==1:
        Y_test.append([1])
    if y_test[i][2]==1:
        Y_test.append([2])

y_test = np.array(Y_test)



#%% Making the Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
print('Accuracy - ', accuracy_score(y_test, pred))


#%%
import seaborn as sns
sns.countplot(x = y_train_new)
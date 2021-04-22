import pickle
import json
import pickle
import pandas as pd
import numpy
import re
import os
import numpy as np
import gensim
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from gensim.models import Doc2Vec
#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
#Questions

train = json.load(open('data/v2_OpenEnded_mscoco_train2014_questions.json'))
val = json.load(open('data/v2_OpenEnded_mscoco_val2014_questions.json'))
test = json.load(open('data/v2_OpenEnded_mscoco_test2015_questions.json'))

train_q = [train['questions'][i]['question'] for i, v in enumerate(train['questions'])]
val_q = [val['questions'][i]['question'] for i, v in enumerate(val['questions'])]
test_q = [test['questions'][i]['question'] for i, v in enumerate(test['questions'])]
total_q = train_q + val_q + test_q
unique_total_q = list(set(total_q))

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(unique_total_q)]

max_epochs = 20
vec_size = 20
alpha = 0.025

#model = Doc2Vec(vector_size=vec_size,
#                alpha=alpha,
#                min_alpha=0.00025,
#                min_count=1,
#                dm=0)
#
#model.build_vocab(tagged_data)
#
#for epoch in range(max_epochs):
#    print('iteration {0}'.format(epoch))
#    model.train(tagged_data,
#                total_examples=model.corpus_count,
#                epochs=model.iter)
#    # decrease the learning rate
#    model.alpha -= 0.0002
#    # fix the learning rate, no decay
#    model.min_alpha = model.alpha
#
#model.save("d2v.model")
#print("Model Saved")

model= Doc2Vec.load("d2v.model")
#to find the vector of a document which is not in training data
#test_data = word_tokenize("I love chatbots".lower())
#v1 = model.infer_vector(test_data)
#print("V1_infer", v1)

x = list(model.docvecs.doctag_syn0)

kmeans = KMeans(n_clusters=3)
kmeans.fit(x)
labels = kmeans.labels_
#train_cluster = {}
#for i in range(len(train['questions'])) :
#    train_cluster['question_id'] = train['questions'][i]['question_id']
#    train['questions'][i]['question_id'] = labels[i]
#val_cluster = {}
#for i in range(len(val['questions'])):
#    val_cluster[i] = val['questions'][i]['question_id']
#
#test_cluster = {}
#for i in range(len(test['questions'])):
#    test_cluster[i] = test['questions'][i]['question_id']
#
#question_type = {}
#for k in range(len(unique_total_q)):
#    question_type[unique_total_q[k]] = labels[k]
#
#with open('data/total_question_type.pkl','wb') as f :
#    pickle.dump(question_type, f)


qtype = pickle.load(open('data/total_question_type.pkl','rb'))
questions = list(qtype.keys())
labels = [qtype[v] for i, v in enumerate(questions)]
import pandas as pd
#pd.DataFrame(labels).describe()
#pd.pivot_table(pd.DataFrame(labels),columns=0)
#labels2 = pd.DataFrame(labels)
#labels2 = labels2.rename(columns={0:'class'})
#labels2.pivot('class')
#labels2.keys()

labels = np.array(labels)
zero = np.where(labels == 0)[0]
one = np.where(labels == 1)[0]
two = np.where(labels == 2)[0]

print(len(zero),len(one),len(two))
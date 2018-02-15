
# coding: utf-8

# # File #1 Feature engineering

# In[ ]:

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from pyspark.sql.functions import isnan, when, count, length, lit, udf, col, struct
import numpy as np
import pyspark.sql.functions as func
import time
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.stem.porter import PorterStemmer
from datetime import datetime
import pandas as pd
from pyspark.sql.functions import length
from pyspark.sql.functions import levenshtein
from pyspark.ml.feature import IDF, Tokenizer, CountVectorizer
from pyspark.sql import SQLContext
spark = SparkSession.builder.getOrCreate()
sqlContext = SQLContext(spark)


# In[2]:

stop_words = nltk.corpus.stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()


# In[3]:

# functions and udfs
def lemmas_nltk(s):
    s_1=s.lower().split()
    s_2=[w for w in s_1 if not w in stop_words]
    s_3=[w for w in s_2 if w.isalpha()]
    s_4=[wordnet_lemmatizer.lemmatize(wordnet_lemmatizer.lemmatize(w,'n'),'v') for w in s_3]
    s_5=' '.join(s_4)
    return(s_5)

def commonNgrams(s1,s2,n):
    s1=lemmas_nltk(s1)
    s2=lemmas_nltk(s2)
    prog=re.compile('([^\s\w]|_)+')
    s1=prog.sub('',s1)
    s2=prog.sub('',s2)
    s1=set(nltk.ngrams(nltk.word_tokenize(s1), n))
    s2=set(nltk.ngrams(nltk.word_tokenize(s2), n))
    return(len(s1 & s2))

def commonNgrams_Q(s1,s2,n):
    prog=re.compile('([^\s\w]|_)+')
    s1=prog.sub('',s1)
    s2=prog.sub('',s2)
    #s1= ' '.join([w for w in s1.lower().split() if not w in stop_words])
    #s2= ' '.join([w for w in s2.lower().split() if not w in stop_words])
    s1=set(nltk.ngrams(nltk.word_tokenize(s1), n))
    s2=set(nltk.ngrams(nltk.word_tokenize(s2), n))
    return(len(s1 & s2))

def wordsCount(my_str):
    return(len(my_str.split()))

def ratio(x,y): return abs(x-y)/(x+y+1e-15) 

def unigram_ratio(ngrams, n1, n2):
    return(ngrams/(1+max(n1,n2)))

def tfidfDist(a,b):
    return float(a.squared_distance(b))


# In[4]:

lemmas_nltk_udf = udf(lemmas_nltk, returnType=StringType())
wordsCount_udf = udf(wordsCount, returnType=IntegerType())
ratio_udf = udf(ratio, DoubleType())
unigram_ratio_udf = udf(unigram_ratio, returnType=DoubleType())
commonNgrams_udf = udf(commonNgrams, returnType=IntegerType())
commonNgrams_Q_udf = udf(commonNgrams_Q, returnType=IntegerType())
dist_udf = udf(tfidfDist, DoubleType())


# In[5]:

trainFileName = "./data/train_sample.csv"
testFileName = "./data/test_sample.csv"


# In[18]:

sch = StructType([StructField('id',IntegerType()),
                  StructField('qid1',IntegerType()),
                  StructField('qid2',IntegerType()),
                  StructField('question1',StringType()),
                  StructField('question2',StringType()),
                  StructField('label',IntegerType())])

train = spark.read.csv(trainFileName, header=True, escape='"', quote='"',schema=sch, multiLine = True)
train = train.drop('qid1', 'qid2')
train = train.dropna()
train.cache()


train=train.withColumn('lemma1', lemmas_nltk_udf(train['question1']))
train=train.withColumn('lemma2', lemmas_nltk_udf(train['question2']))

for i in ["1","2"]:
    train = train.withColumn('lWCount'+i, wordsCount_udf(train['lemma'+i]))
    train = train.withColumn('qWCount'+i, wordsCount_udf(train['question'+i]))
    train = train.withColumn('lLen'+i, length(train['lemma'+i]))
    train = train.withColumn('qLen'+i, length(train['question'+i]))

train = train.withColumn('lWCount_ratio', ratio_udf(train['lWCount1'],train['lWCount2']))
train = train.withColumn('qWCount_ratio', ratio_udf(train['qWCount1'],train['qWCount2']))
train = train.withColumn('lLen_ratio', ratio_udf(train['lLen1'],train['lLen2']))
train = train.withColumn('qLen_ratio', ratio_udf(train['qLen1'],train['qLen2']))

for i in ['1','2','3']:
    train = train.withColumn('ngram', lit(int(i)))
    train = train.withColumn('lNgrams_'+i,commonNgrams_udf(train['lemma1'],train['lemma2'],train['ngram']))
    train = train.withColumn('qNgrams_'+i,commonNgrams_Q_udf(train['question1'],train['question2'],train['ngram']))
    train = train.drop('ngram')

train = train.withColumn('lUnigram_ratio', unigram_ratio_udf(train['lNgrams_1'],train['lWCount1'],train['lWCount2']))
train = train.withColumn('qUnigram_ratio', unigram_ratio_udf(train['qNgrams_1'],train['qWCount1'],train['qWCount2']))

tokenizer = Tokenizer(inputCol="lemma1", outputCol="words1")
train = tokenizer.transform(train)

tokenizer.setParams(inputCol="lemma2", outputCol="words2")
train = tokenizer.transform(train)

corpus = train.selectExpr('words1 as words').join(train.selectExpr('words2 as words'), on='words', how='full')
cv = CountVectorizer(inputCol="words", outputCol="tf", minDF=2.0)

cvModel = cv.fit(corpus)
corpus = cvModel.transform(corpus)

rest1=train.select(['id','question1'])
rest2=train.select(['id','question2'])
tokenizer = Tokenizer(inputCol="question1", outputCol="words1")
rest1 = tokenizer.transform(rest1)
tokenizer.setParams(inputCol="question2", outputCol="words2")
rest2 = tokenizer.transform(rest2)

res1 = cvModel.transform(rest1.selectExpr('id', 'words1 as words'))
res2 = cvModel.transform(rest2.selectExpr('id', 'words2 as words'))

idf = IDF(inputCol="tf", outputCol="idf")
idfModel = idf.fit(corpus)
res1 = idfModel.transform(res1)
res2 = idfModel.transform(res2)

res = res1.selectExpr('id','idf as idf1').join(res2.selectExpr('id','idf as idf2'), on='id', how='inner')

res = res.withColumn('dist', dist_udf(res['idf1'], res['idf2']))

train = train.drop('words1', 'words2')
train = train.join(res.selectExpr('id','dist as tfidfDistance'),on='id',how='inner')

train = train.withColumn('lLeven', levenshtein(train['lemma1'],train['lemma2']))
train = train.withColumn('qLeven', levenshtein(train['question1'],train['question2']))

train=train.select('id','lWCount1','qWCount1','lLen1','qLen1','lWCount2','qWCount2','lLen2','qLen2','lWCount_ratio','qWCount_ratio','lLen_ratio','qLen_ratio','lNgrams_1','qNgrams_1','lNgrams_2','qNgrams_2','lNgrams_3','qNgrams_3',                  'lUnigram_ratio','qUnigram_ratio','tfidfDistance','lLeven','qLeven','label')

train.coalesce(1).write.csv('./results_local/results_train_1000/train_features.csv', sep='\t')


# &nbsp;
# 
# &nbsp;
# 
# &nbsp;
# 

# In[21]:

test_sch = StructType([StructField('test_id',IntegerType()),
                       StructField('question1',StringType()),
                       StructField('question2',StringType())])

test = spark.read.csv(testFileName, header=True, escape='"',quote='"',schema=test_sch, multiLine = True)
test = test.dropna()
test.cache()

test=test.withColumn('lemma1', lemmas_nltk_udf(test['question1']))
test=test.withColumn('lemma2', lemmas_nltk_udf(test['question2']))
for i in ["1","2"]:
    test = test.withColumn('lWCount'+i, wordsCount_udf(test['lemma'+i]))
    test = test.withColumn('qWCount'+i, wordsCount_udf(test['question'+i]))
    test = test.withColumn('lLen'+i, length(test['lemma'+i]))
    test = test.withColumn('qLen'+i, length(test['question'+i]))
test = test.withColumn('lWCount_ratio', ratio_udf(test['lWCount1'],test['lWCount2']))
test = test.withColumn('qWCount_ratio', ratio_udf(test['qWCount1'],test['qWCount2']))
test = test.withColumn('lLen_ratio', ratio_udf(test['lLen1'],test['lLen2']))
test = test.withColumn('qLen_ratio', ratio_udf(test['qLen1'],test['qLen2']))

for i in ['1','2','3']:
    test = test.withColumn('ngram', lit(int(i)))
    test = test.withColumn('lNgrams_'+i,commonNgrams_udf(test['lemma1'],test['lemma2'],test['ngram']))
    test = test.withColumn('qNgrams_'+i,commonNgrams_Q_udf(test['question1'],test['question2'],test['ngram']))
    test = test.drop('ngram')

test = test.withColumn('lUnigram_ratio', unigram_ratio_udf(test['lNgrams_1'],test['lWCount1'],test['lWCount2']))
test = test.withColumn('qUnigram_ratio', unigram_ratio_udf(test['qNgrams_1'],test['qWCount1'],test['qWCount2']))

tokenizer = Tokenizer(inputCol="lemma1", outputCol="words1")
test = tokenizer.transform(test)

tokenizer.setParams(inputCol="lemma2", outputCol="words2")
test = tokenizer.transform(test)

corpus = test.selectExpr('words1 as words').join(test.selectExpr('words2 as words'), on='words', how='full')
cv = CountVectorizer(inputCol="words", outputCol="tf", minDF=2.0)

cvModel = cv.fit(corpus)
corpus = cvModel.transform(corpus)

rest1=test.select(['test_id','question1'])
rest2=test.select(['test_id','question2'])

tokenizer = Tokenizer(inputCol="question1", outputCol="words1")
rest1 = tokenizer.transform(rest1)
tokenizer.setParams(inputCol="question2", outputCol="words2")
rest2 = tokenizer.transform(rest2)

res1 = cvModel.transform(rest1.selectExpr('test_id', 'words1 as words'))
res2 = cvModel.transform(rest2.selectExpr('test_id', 'words2 as words'))

idf = IDF(inputCol="tf", outputCol="idf")
idfModel = idf.fit(corpus)
res1 = idfModel.transform(res1)
res2 = idfModel.transform(res2)

res = res1.selectExpr('test_id','idf as idf1').join(res2.selectExpr('test_id','idf as idf2'), on='test_id', how='inner')

res = res.withColumn('dist', dist_udf(res['idf1'], res['idf2']))

test = test.drop('words1', 'words2')

test = test.join(res.selectExpr('test_id','dist as tfidfDistance'),on='test_id',how='inner')

test = test.withColumn('lLeven', levenshtein(test['lemma1'],test['lemma2']))
test = test.withColumn('qLeven', levenshtein(test['question1'],test['question2']))

test=test.select('test_id','lWCount1','qWCount1','lLen1','qLen1','lWCount2','qWCount2','lLen2','qLen2','lWCount_ratio','qWCount_ratio','lLen_ratio','qLen_ratio','lNgrams_1','qNgrams_1','lNgrams_2','qNgrams_2','lNgrams_3','qNgrams_3',                  'lUnigram_ratio','qUnigram_ratio','tfidfDistance','lLeven','qLeven')

test.coalesce(1).write.csv('./results_local/results_test_1000/test_features.csv', sep='\t')


# In[10]:

train.printSchema()


# In[17]:

test.printSchema()


# In[ ]:




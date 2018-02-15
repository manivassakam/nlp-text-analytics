
# coding: utf-8

# # File #2 model fitting

# In[104]:

reset


# In[1]:

from pyspark.sql import SparkSession
import pyspark.sql.functions as func
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType, DoubleType
from pyspark.sql.functions import isnan, when, count, length, lit, udf, col, struct
from math import log
import pandas as pd
from pyspark.sql import DataFrame
from functools import reduce
from sklearn.metrics import log_loss
import itertools
from pyspark.sql import SQLContext
spark = SparkSession.builder.getOrCreate()
sqlContext = SQLContext(spark)


# In[2]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[3]:

def unionAll(*dfs): return reduce(DataFrame.unionAll, dfs)

def expandgrid(*itrs):                                             
   product = list(itertools.product(*itrs))
   return {'Var{}'.format(i+1):[x[i] for x in product] for i in range(len(itrs))}


prob_of_one_udf = func.udf(lambda v: float(v[1]), FloatType())


# In[4]:

sch = StructType([StructField('id',IntegerType()),                   StructField('lWCount1',IntegerType()),                  StructField('qWCount1',IntegerType()),                  StructField('lLen1',IntegerType()),                  StructField('qLen1',IntegerType()),                  StructField('lWCount2',IntegerType()),                  StructField('qWCount2',IntegerType()),                  StructField('lLen2',IntegerType()),                  StructField('qLen2',IntegerType()),                  StructField('lWCount_ratio',DoubleType()),                  StructField('qWCount_ratio',DoubleType()),                  StructField('lLen_ratio',DoubleType()),                  StructField('qLen_ratio',DoubleType()),                  StructField('lNgrams_1',IntegerType()),                  StructField('qNgrams_1',IntegerType()),                  StructField('lNgrams_2',IntegerType()),                  StructField('qNgrams_2',IntegerType()),                  StructField('lNgrams_3',IntegerType()),                  StructField('qNgrams_3',IntegerType()),                  StructField('lUnigram_ratio',DoubleType()),                  StructField('qUnigram_ratio',DoubleType()),                  StructField('tfidfDistance',DoubleType()),                  StructField('lLeven',IntegerType()),                  StructField('qLeven',IntegerType()),                  StructField('label',IntegerType())])


# In[5]:

sch_test = StructType([StructField('id',IntegerType()),                   StructField('lWCount1',IntegerType()),                  StructField('qWCount1',IntegerType()),                  StructField('lLen1',IntegerType()),                  StructField('qLen1',IntegerType()),                  StructField('lWCount2',IntegerType()),                  StructField('qWCount2',IntegerType()),                  StructField('lLen2',IntegerType()),                  StructField('qLen2',IntegerType()),                  StructField('lWCount_ratio',DoubleType()),                  StructField('qWCount_ratio',DoubleType()),                  StructField('lLen_ratio',DoubleType()),                  StructField('qLen_ratio',DoubleType()),                  StructField('lNgrams_1',IntegerType()),                  StructField('qNgrams_1',IntegerType()),                  StructField('lNgrams_2',IntegerType()),                  StructField('qNgrams_2',IntegerType()),                  StructField('lNgrams_3',IntegerType()),                  StructField('qNgrams_3',IntegerType()),                  StructField('lUnigram_ratio',DoubleType()),                  StructField('qUnigram_ratio',DoubleType()),                  StructField('tfidfDistance',DoubleType()),                  StructField('lLeven',IntegerType()),                  StructField('qLeven',IntegerType())])


# ## Local run

# In[6]:

trainFileName='./results_local/results_train_1000/train_features.csv/*.csv'
testFileName='./results_local/results_test_1000/test_features.csv/*.csv' 


# In[7]:

train=spark.read.csv(trainFileName, header=True, sep='\t', schema=sch, escape='"', quote='"',multiLine = False)
test=spark.read.csv(testFileName, header=False, sep='\t', schema=sch_test, escape='"', quote='"',multiLine = False)


# In[8]:

Features=['lWCount1','qWCount1','lLen1','qLen1','lWCount2','qWCount2','qLen2','qLen2',          'lWCount_ratio', 'qWCount_ratio','lLen_ratio', 'qLen_ratio',          'lNgrams_1', 'qNgrams_1', 'lNgrams_2', 'qNgrams_2','lNgrams_3','qNgrams_3',          'lUnigram_ratio','qUnigram_ratio','tfidfDistance','lLeven','qLeven']

assembler=VectorAssembler(inputCols=Features, outputCol='features')
train=assembler.transform(train)
train=train.select('id','features','label')

(train_split, test_split)=train.randomSplit([0.7, 0.3], seed=165)

train_split.cache()
test_split.cache()


# In[9]:

test=assembler.transform(test)
test=test.select('id','features')
test.cache()


# In[10]:

train_split.show(10)


# In[11]:

test_split.show(10)


# In[12]:

test.show(10)


# Initiate `RandomForestClassifier` class from `pyspark.ml.classification` module. Sintax is:  
#      
# `RandomForestClassifier(
#     featuresCol='features',
#     labelCol='label',
#     predictionCol='prediction',
#     probabilityCol='probability',
#     rawPredictionCol='rawPrediction',
#     maxDepth=5, 
#     maxBins=32, 
#     minInstancesPerNode=1, 
#     minInfoGain=0.0, 
#     maxMemoryInMB=256, 
#     cacheNodeIds=False, 
#     checkpointInterval=10, 
#     impurity='gini', 
#     numTrees=20, 
#     featureSubsetStrategy='auto', 
#     seed=None, 
#     subsamplingRate=1.0)`
# * numTrees - Number of trees to train (>= 1).
# * featureSubsetStrategy - Fraction of the training data used for learning each decision tree, in range (0, 1]. supportedFeatureSubsetStrategies = {'auto', 'all', 'onethird', 'sqrt', 'log2'}

# In[17]:

glr=GeneralizedLinearRegression(family="binomial",link="logit",featuresCol="features",labelCol="label")
logitModel=glr.fit(train_split)
logitPredictions=logitModel.transform(test_split)
evaluator=BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
glm_AUC=evaluator.evaluate(logitPredictions)

# now estimat the model using the entire training set
logitModel_full=glr.fit(train)
test_glm=logitModel_full.transform(test)
outdf_glm=test_glm.withColumn('predict', func.round(test_glm['prediction'],6)).select('id','predict')
val_glm=round(glm_AUC,5)


# In[18]:

AUC_list=[]
for i in range(len(mygrid.index)):
    (a,b,c)=mygrid.iloc[i]
    
    rf_classifier=RandomForestClassifier(maxDepth=a,numTrees=b,maxBins=c,maxMemoryInMB=10000,featureSubsetStrategy='auto')
    rfModel=rf_classifier.fit(train_split)
    
    rf=rfModel.transform(test_split)
    evaluator=BinaryClassificationEvaluator()
    rf_AUC=evaluator.evaluate(rf)
    AUC_list.append(rf_AUC)


# In[19]:

mygrid['meanAUC']=AUC_list  
mygrid


# In[20]:

mygrid.iloc[mygrid.idxmax(axis=0)[3]]


# In[50]:

imp = pd.Series(rfModel.featureImportances.toArray(), index=Features)
imp.sort_values(inplace=True)
#plt.figure(figsize=(20,15))
#imp[-30:].plot(kind = 'barh',title='Features Importances in Random Forest Model')


# Create `GBTClassifier` class from `pyspark.ml.classification` module. Syntax is:   
#     
# `GBTClassifier(
#     featuresCol='features',
#     labelCol='label',
#     predictionCol='prediction',
#     maxDepth=5, 
#     maxBins=32, 
#     minInstancesPerNode=1,
#     minInfoGain=0.0,
#     maxMemoryInMB=256,
#     cacheNodeIds=False,
#     checkpointInterval=10, 
#     lossType='logistic',
#     maxIter=20,
#     stepSize=0.1,
#     seed=None,
#     subsamplingRate=1.0)`
# * lossType - Loss function which GBT tries to minimize (case-insensitive). Supported options: logistic.
# * maxIter - max number of iterations (>= 0).
# * stepSize - Step size to be used for each iteration of optimization (>= 0).

# In[47]:

maxDepth=[6,8]
maxIter=[17,18,19]
stepSize=[0.1]
maxBins=[20]
gbt_mygrid=pd.DataFrame(expandgrid(maxDepth,maxIter,stepSize,maxBins))    
gbt_mygrid


# In[48]:

gbt_AUC_list=[]
for i in range(len(gbt_mygrid.index)):
    (a,b,c,d)=gbt_mygrid.iloc[i]
    
    gbt_classifier=GBTClassifier(maxDepth=a, maxIter=b, stepSize=c,maxBins=d,maxMemoryInMB=10000)
    gbtModel=gbt_classifier.fit(train_split)
    gbt=gbtModel.transform(test_split)
    evaluator=BinaryClassificationEvaluator()
    gbt_AUC=evaluator.evaluate(gbt)
    gbt_AUC_list.append(gbt_AUC)

gbt_mygrid['gbt_AUC']=gbt_AUC_list 
gbt_mygrid


# In[46]:

gbt_mygrid


# In[49]:

gbt_mygrid.iloc[gbt_mygrid.idxmax(axis=0)[4]]


# In[51]:

imp = pd.Series(gbtModel.featureImportances.toArray(), index=Features)
imp.sort_values(inplace=True)
#plt.figure(figsize=(20,15))
#imp[-30:].plot(kind = 'barh',title='Features Importances in GBT Model');


# In[60]:

test=gbtModel.transform(test)
outdf = test.withColumn('predict', func.round(prob_of_one_udf(test['probability']),6)).select('id','predict')


# In[ ]:

# local test run save
val=round(gbt_AUC,5)
outdf.coalesce(1).write.csv('./results/05_prediction/test_predictions_{}.csv'.format(val), sep='\t')


# In[ ]:

# rcc save
val=round(gbt_AUC,5)
outdf.coalesce(1).write.csv('./results_model/test_predictions_{}_.csv'.format(val), sep='\t')


# ## RCC run

# In[47]:

#rcc file location
trainFileName = "./results_train/train_features.csv/*.csv"
test_loc_1='./results_test_part_1/test_features.csv/*.csv'
test_loc_2='./results_test_part_2/test_features.csv/*.csv'


# In[ ]:

# for rcc run
train=spark.read.csv(trainFileName, header=True, sep='\t', schema=sch, escape='"', quote='"',multiLine = False)
test_1=spark.read.csv(test_loc_1, header=False, sep='\t', schema=sch_test, escape='"', quote='"',multiLine = False)
test_2=spark.read.csv(test_loc_2, header=False, sep='\t', schema=sch_test, escape='"', quote='"',multiLine = False)


# In[ ]:

Features=['lWCount1','qWCount1','lLen1','qLen1','lWCount2','qWCount2','qLen2','qLen2',          'lWCount_ratio', 'qWCount_ratio','lLen_ratio', 'qLen_ratio',          'lNgrams_1', 'qNgrams_1', 'lNgrams_2', 'qNgrams_2','lNgrams_3','qNgrams_3',          'lUnigram_ratio','qUnigram_ratio','tfidfDistance']

assembler=VectorAssembler(inputCols=Features, outputCol='features')
train=assembler.transform(train)
train=train.select('id','features','label')

(train_split, test_split)=train.randomSplit([0.7, 0.3], seed=165)

train_split.cache()
test_split.cache()


# In[ ]:

#skip in local run (only or rcc or when the test_features are split)
test_1=assembler.transform(test_1)
test_1=test_1.select('id','features')
test_2=assembler.transform(test_2)
test_2=test_2.select('id','features')

test_1=test_1.select('id','features')
test_2=test_2.select('id','features')
test=unionAll(test_1,test_2)


# In[ ]:

test=assembler.transform(test)
test=test.select('id','features')
test.cache()


# In[ ]:

glr=GeneralizedLinearRegression(family="binomial",link="logit",featuresCol="features",labelCol="label")
logitModel=glr.fit(train_split)
logitPredictions=logitModel.transform(test_split)
evaluator=BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
glm_AUC=evaluator.evaluate(logitPredictions)

# now estimat the model using the entire training set
logitModel_full=glr.fit(train)
test_glm=logitModel_full.transform(test)
outdf_glm=test_glm.withColumn('predict', func.round(test_glm['prediction'],6)).select('id','predict')
val_glm=round(glm_AUC,5)


# In[ ]:

rf_classifier=RandomForestClassifier(maxDepth=9,numTrees=1500,maxBins=7,featureSubsetStrategy='auto') # maxMemoryInMB=10000,
rfModel=rf_classifier.fit(train_split)
rf=rfModel.transform(test_split)
evaluator=BinaryClassificationEvaluator()
rf_AUC=evaluator.evaluate(rf)

# now estimat the model using the entire training set
rfModel_full=rf_classifier.fit(train)
test_rf=rfModel_full.transform(test)
outdf_rf = test_rf.withColumn('predict', func.round(prob_of_one_udf(test_rf['probability']),6)).select('id','predict')
val_rf=round(rf_AUC,5)


# In[ ]:

gbt_classifier= GBTClassifier(maxDepth=8, maxIter=19, stepSize=.1,maxBins=20)#,maxMemoryInMB=10000)
gbtModel = gbt_classifier.fit(train_split)
gbt=gbtModel.transform(test_split)
evaluator = BinaryClassificationEvaluator()
gbt_AUC = evaluator.evaluate(gbt)

# now estimat the model using the entire training set
gbtModel_full=gbt_classifier.fit(train)
test_gbt=gbtModel_full.transform(test)
outdf_gbt= test_gbt.withColumn('predict', func.round(prob_of_one_udf(test_gbt['probability']),6)).select('id','predict')
val_gbt=round(gbt_AUC,5)


# In[ ]:

outdf_glm.orderBy('id').coalesce(1).write.csv('./results_model/glm/test_predictions_{}_.csv'.format(val_glm), sep='\t')


# In[ ]:

outdf_rf.orderBy('id').coalesce(1).write.csv('./results_model/rf/test_predictions_{}_.csv'.format(val_rf), sep='\t')


# In[ ]:

outdf_gbt.orderBy('id').coalesce(1).write.csv('./results_model/gbt/test_predictions_{}_.csv'.format(val_gbt), sep='\t')


# &nbsp;
# 
# &nbsp;
# 

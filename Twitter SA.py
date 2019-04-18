# Databricks notebook source
# DBTITLE 1,Importing Libraries
# MAGIC %md Importing important libraries
# MAGIC The libraries include:
# MAGIC pandas - data manipulation and wrangling
# MAGIC nltk - natural language processessing
# MAGIC sklearn - feature extraction
# MAGIC bs4 - html handling
# MAGIC re, string, itertools - Data manipulation
# MAGIC langid - Language detection.

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer
from nltk import word_tokenize,WordNetLemmatizer,TweetTokenizer,PorterStemmer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import nltk
from bs4 import BeautifulSoup as bs
import re
import string
from itertools import groupby
import langid
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# COMMAND ----------

df =spark.sql("Select int(Sentiment) as label, tweet from training_twitter_shuff")
df.count()

# COMMAND ----------

# DBTITLE 1,Initial Data Exploration
# MAGIC %md converting spark dataframe to pandas df for easy initial data exploration

# COMMAND ----------

pd_df=df.toPandas()
pd_df.head()

# COMMAND ----------

pd_df['Len']=[len(i) for i in pd_df['tweet']]
pd_df.Len.plot(kind='box', by ='label')
display()

# COMMAND ----------

# MAGIC %md Tweeter character limit was 140 during the time of data collection, therefore a character length more than 140 indicate issues in data

# COMMAND ----------

sample=pd_df[pd_df['Len']>140].sort_values('Len', ascending=False).tweet
sample

# COMMAND ----------

# DBTITLE 1,Data Cleaning
# MAGIC %md Python functions are created to clean the tweets to final modeling data.
# MAGIC 
# MAGIC htmlcleaning - converts html formatted tweets to actual tweets. ||
# MAGIC tag_and_remove - keeps only nouns, adjectives, verbs and adverbs only in the tweets. ||
# MAGIC lemmatize - converts words to its lemmatized form. ||
# MAGIC check_lang - predict the language of the given text.

# COMMAND ----------

def htmlcleaning(data_str):
  return bs(data_str,"lxml").get_text()


def tag_and_remove(data_str):
    nltk.download('averaged_perceptron_tagger')
    cleaned_str = ' '
    # noun tags
    nn_tags = ['NN', 'NNP', 'NNP', 'NNPS', 'NNS']
    # adjectives
    jj_tags = ['JJ', 'JJR', 'JJS']
    # verbs
    vb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    # adverbs
    avb_tags =['RB','RBR','RBS']
    
    nltk_tags = nn_tags + jj_tags + vb_tags + avb_tags
    # break string into 'words'
    text = data_str.split()
    # tag the text and keep only those with the right tags
    tagged_text = pos_tag(text)
    #print(tagged_text)
    for tagged_word in tagged_text:
        if tagged_word[1] in nltk_tags:
            cleaned_str += tagged_word[0] + ' '
    return cleaned_str
  
  
def lemmatize(data_str):
    nltk.download('wordnet')
    # expects a string
    list_pos = 0
    cleaned_str = ''
    lmtzr = WordNetLemmatizer()
    text = data_str.split()
    tagged_words = pos_tag(text)
    for word in tagged_words:
        if 'v' in word[1].lower():
            lemma = lmtzr.lemmatize(word[0], pos='v')
        else:
            lemma = lmtzr.lemmatize(word[0], pos='n')
        if list_pos == 0:
            cleaned_str = lemma
        else:
            cleaned_str = cleaned_str + ' ' + lemma
        list_pos += 1
    return cleaned_str
  
  
def check_lang(data_str):
    predict_lang = langid.classify(data_str)
    if predict_lang[1] >= .9:
        language = predict_lang[0]
    else:
        language = 'NA'
    return language

# COMMAND ----------

# MAGIC %md cleaning_irrelevant - cleaning irrelevant words from the tweets 
# MAGIC 1. handing tweeting word (words with repeated characters usually used in tweets)
# MAGIC 2. handling apostrophes words
# MAGIC 3. removing urls and hyperlinks
# MAGIC 4. handling punctuations
# MAGIC 5. handling numbers and alphanumeric words
# MAGIC 6. remove @callouts and #mentions

# COMMAND ----------

def cleaning_irrelevant(data_str):
  #handling tweeting word
  commonBigrams=['l','s','e','o','t','f','p','r','m','c','n','d','g','i','b']
  data_str=''.join(''.join(i)[:2] if value in commonBigrams else ''.join(i)[:1] for value,i in groupby(data_str))
  #handling aphostrophe words
  apostrophes={"i'm":"i am","aren't":"are not","can't":"cannot","couldn't":"could not","didn't":"did not","doesn't":"does not","don't":"do not","hadn't":"had not","hasn't":"has not","haven't":"have not","he'd":"he had","he'll":"he will","he's":"he is","I'd":"I had","I'll":"I will","I'm":"I am","I've":"I have","isn't":"is not","it's":"it is","let's":"let us","mustn't":"must not","shan't":"shall not","she'd":"she had","she'll":"she will","she's":"she is","shouldn't":"should not","that's":"that is","there's":"there is","they'd":"they had","they'll":"they will","they're":"they are","they've":"they have","we'd":"we had","we're":"we are","we've":"we have","weren't":"were not","what'll":"what will","what're":"what are","what's":"what is","what've":"what have","where's":"where is","who'd":"who had","who'll":"who will","who're":"who are","who's":"who is","who've":"who have","won't":"will not","wouldn't":"would not","you'd":"you had","you'll":"you will","you're":"you are","you've":"you have","y'll":"you all"}
  data_str=' '.join(apostrophes[word.lower()] if word.lower() in apostrophes else word for word in data_str.split())
  #handling urls
  url_re = re.compile('https?://(www.)?\w+\.\w+(/\w+)*/?')
  #handling callouts & Hashtags
  mention_re = re.compile('[@#](\w+)')
  #handling punctuations
  punc_re = re.compile('[%s]' %re.escape(string.punctuation))
  #handling numbers
  num_re = re.compile('(\\d+)')
  #handling valid words
  alpha_num_re = re.compile("^[a-z0-9_.]+$")
  #stop words
  #stop_words=set(stopwords.words('english'))
  # convert to lowercase
  data_str = data_str.lower()
  # remove hyperlinks
  data_str = url_re.sub(' ', data_str)
  # remove @mentions
  data_str = mention_re.sub(' ', data_str)
  # remove puncuation
  data_str = punc_re.sub(' ', data_str)
  # remove numeric 'words'
  data_str = num_re.sub(' ', data_str)
  data_str=' '.join(word for word in data_str.split() if alpha_num_re.match(word)) #and not word in stop_words)
  return data_str

# COMMAND ----------

for tweet in mini:
  print(lemmatize(tag_and_remove(cleaning_irrelevant(htmlcleaning(tweet)))))

# COMMAND ----------

# DBTITLE 1,Converting Python functions to Pyspark functions
# MAGIC %md converting python functions to pyspark udfs using udf and stringtype functions

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# COMMAND ----------

remove_features_udf = udf(cleaning_irrelevant, StringType())
tag_and_remove_udf = udf(tag_and_remove, StringType())
lemmatize_udf = udf(lemmatize, StringType())
htmlcleaning_udf = udf(htmlcleaning, StringType())
check_lang_udf=udf(check_lang,StringType())

# COMMAND ----------

from pyspark.sql.functions import length
clean_df=df.withColumn("clean",lemmatize_udf(tag_and_remove_udf(remove_features_udf(htmlcleaning_udf((df['tweet']))))))
clean_df_2 = clean_df.select('label','clean').filter(length(clean_df.clean)>0)
clean_df_2.count()

# COMMAND ----------

# Split the data into training and test sets (40% held out for testing)
(trainingData, testData) = clean_df_2.randomSplit([0.6, 0.4])

# COMMAND ----------

# DBTITLE 1,Creating ML pipeline in Pyspark
# MAGIC %md Creating pipeline using tokenizer | ngram | combiner | TF-IDF | LogisticRegression

# COMMAND ----------

from pyspark.ml.feature import HashingTF, IDF, Tokenizer,NGram,VectorAssembler,SQLTransformer
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier,LogisticRegression
from pyspark.sql.functions import col, concat 
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator

# Configure an ML pipeline, which consists of tree stages: tokenizer, hashingTF, and nb.
tokenizer = Tokenizer(inputCol="clean", outputCol="word")

ngram = NGram(n=2, inputCol="word", outputCol="ngrams")

sql = SQLTransformer(statement="SELECT *, concat(word,ngrams) AS word_comb FROM __THIS__")

hashingTF = HashingTF(inputCol="word_comb", outputCol="rawFeatures")

idf = IDF(minDocFreq=3, inputCol="rawFeatures", outputCol="features")

# Logistic regression model
logistic = LogisticRegression(maxIter=10,regParam=0.001,elasticNetParam=1)

# Pipeline Architecture

pipeline=Pipeline(stages=[
    tokenizer, 
    ngram,
    sql,
    hashingTF,
    idf,
    logistic
])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# COMMAND ----------

# DBTITLE 1,Tuning Model
# MAGIC %md Tuning logistic model using grid search method. Paramter grid is created with various values of regularisation threshold and Regularizition methods (Lasso, Ridge, Elastic Net)

# COMMAND ----------

paramGrid = ParamGridBuilder().addGrid(logistic.regParam, [0.0001,0.001,0.01,0.1,1,10]).addGrid(logistic.elasticNetParam,[0,0.1,0.9,1]).build()
cv = CrossValidator(estimator=pipeline, evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid)

cvModel = cv.fit(trainingData)

# COMMAND ----------

model = cvModel.bestModel

# COMMAND ----------

# DBTITLE 1,Model Evaluation
predictions = model.transform(testData)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
evaluator.evaluate(predictions)

# COMMAND ----------

extract= predictions.select("label","prediction").toPandas()
extract.head()

# COMMAND ----------

predictions = model.transform(testData)
#predictions.show()
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator.evaluate(predictions)

# COMMAND ----------

# DBTITLE 1,Testing Model
mylist=["Heading to the not Great State of Minnesota!"]
test=spark.createDataFrame(mylist, StringType())
test=test.selectExpr("value as tweet")

test_df=test.withColumn("clean",lemmatize_udf(tag_and_remove_udf(remove_features_udf(htmlcleaning_udf(test['tweet'])))))

#pred_test.show()

# COMMAND ----------

pred_test = model.transform(test_df)
selected = pred_test.select("tweet","clean", "probability", "prediction")
for row in selected.collect():
    tweet, clean, prob, prediction = row
    if prediction == 4:
      pred = "Positive"
    else:
      pred = "Negative"
    print("%s -->%s-->prediction=%s\n" % (tweet,clean,pred))

# COMMAND ----------

# DBTITLE 1,Exporting Model 
# MAGIC %md Exporting and importing model using mleap library

# COMMAND ----------

model.save("/FileStore/spark-logistic-regression-model")

# COMMAND ----------

from pyspark.ml import PipelineModel
sameModel = PipelineModel.load("/FileStore/spark-logistic-regression-model")
mylist=["Heading to the not Great State of Minnesota!"]
test=spark.createDataFrame(mylist, StringType())
test=test.selectExpr("value as tweet")

test_df=test.withColumn("clean",lemmatize_udf(tag_and_remove_udf(remove_features_udf(htmlcleaning_udf(test['tweet'])))))
pred_test = sameModel.transform(test_df)
selected = pred_test.select("tweet","clean", "probability", "prediction")
for row in selected.collect():
    tweet, clean, prob, prediction = row
    if prediction == 4:
      pred = "Positive"
    else:
      pred = "Negative"
    print("%s -->%s-->prediction=%s\n" % (tweet,clean,pred))

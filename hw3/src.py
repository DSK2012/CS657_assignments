from __future__ import print_function
from cmath import nan
from os import truncate

from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import IntegerType,StringType
from pyspark.sql.functions import col,isnan,when,count,lower,regexp_replace,udf
from pyspark.ml.feature import Imputer,Tokenizer,StopWordsRemover,HashingTF, IDF, CountVectorizer, VectorAssembler, RegexTokenizer
from pyspark.ml.classification import LogisticRegression,RandomForestClassifier,MultilayerPerceptronClassifier,LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator,BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from langdetect import detect
from textblob import TextBlob



import nltk
from nltk.corpus import stopwords
import re
stop_words = set(stopwords.words('english'))
stop_words.add(' ')
stop_words.add('')

def lower_punc(data):
    # print(data)
    # print("!"*100)
    # data = str(data)
    # print(data)
    try:
        data = data.replace('\n',' ')
        data = data.replace('\\','')
        punc='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        lower_line = data.lower()
        for c in punc:
            lower_line = lower_line.replace(c,' ')
        re.sub(re.compile('<.*?>'),'',lower_line)
        # lower_line = re.sub(r'[0-9]', '', lower_line)
    except:
        lower_line = data
    return lower_line

def remove_stop_words(data):
    # print(data)
    try:
        new_data = [word for word in data if word not in stop_words]
    except:
        new_data = data
    return new_data



def main():
    spark = SparkSession.builder.appName('hw3').getOrCreate()
    data = spark.read.csv("hashtag_joebiden.csv", sep=',', multiLine=True,header = True)

    # cols = data.select([count(when(col(c).contains("\n"),c)) for c in data.columns])
    # count= data.select([count(when(col(c).isNull(), c)).alias(c) for c in data.columns] ).first()
    # print(cols)
    tweet=data.select(col("tweet"))
    print(type(tweet))
    # counts =tweet.select([count(when(col(c).isNull(), c)).alias(c) for c in data.columns] ).first()
    # print(counts)
    # print(data.count())

    tweet=tweet.na.drop()
    print(tweet.count())

   



    def detect_english(text):
        try:
            return detect(text)=='en'
        except:
            return 'none'
    detectUDF= udf(lambda x : detect_english(x),StringType())
    tweet1=tweet.withColumn('lang',detectUDF(col("tweet")))

    tweet1 = tweet1.filter((tweet1.lang == 'true'))
    print(tweet1.count())
    tweet1.write.option("header",True).csv("eng_data.csv")
    # tweet1= tweet1.filter((tweet1.lang!='true'))

    # tweet1.show()
    # print(tweet1.count())

#     def sentiment_analysis(tweet):
#     # Determine polarity
#         polarity = TextBlob(" ".join(tweet)).sentiment.polarity
#     # Classify overall sentiment
#         if polarity > 0:
#         # positive
#             sentiment = 1
#         elif polarity == 0:
#         # neutral
#             sentiment = 2
#         else:
#         # negative
#              sentiment = 2
#         return sentiment

#     polarityUDF= udf(lambda x: sentiment_analysis(x),StringType())
#     #tweet=tweet.select(polarityUDF(col("tweet")).alias("tweet") )
#     tweet=tweet.withColumn("sentiment", polarityUDF("tweet"))
#     tweet.show(5)

#     tweet = tweet.select((lower(regexp_replace('tweet', "[^a-zA-Zd+ ]", "")).alias('tweet')))
    
#     # tweet = tweet.select((lower(regexp_replace('tweet', "\s+", "")).alias('tweet')))

#     # tweet = tweet.filter((data.fraudulent == 0) | (data.fraudulent == 1))
    
#     tokenizer = Tokenizer(inputCol="tweet", outputCol="tweet1")
#     data = tokenizer.transform(tweet)
#     remover = StopWordsRemover(inputCol="tweet1", outputCol="tweet2")
#     data=remover.transform(data)
    
#     # reg_tokenizer = RegexTokenizer(inputCol = "tweet2",outputCol = "regout",pattern = "/[a0-z9]/g")
#     # data = reg_tokenizer.transform(data)

#    # print(tweets_rdd.collect()[0])
#     # tweets = tweets_rdd.toDF()
#     # tweets.show()
#     # print(tweets.count())
#     # print(data.count())
#     # data = data.union(tweets)

#     data.show()


#     # data.show()



if __name__ == "__main__":
    main()
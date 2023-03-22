#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from pyspark import SparkConf, SparkContext
from math import sqrt
from itertools import islice
import sys
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import *
from pyspark.sql.functions import col,isnan,when,count
# from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator, RankingEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
# from pyspark.ml.evaluation import RankingMetrics
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import *
from pyspark.sql.functions import col,split,isnan,when,count,count,lower,regexp_replace
from pyspark.ml.feature import Imputer,Tokenizer,StopWordsRemover,HashingTF, IDF, CountVectorizer, VectorAssembler
from pyspark.ml.regression import LinearRegression
# from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.types import IntegerType


# In[2]:


conf = SparkConf()
sc = SparkContext(conf = conf)


# In[42]:


def loadMovieNames():
    movieNames = {}
    with open("ml-20m/movies.csv", encoding="ISO-8859-1") as f:
        for line in f:
            fields = line.split("|")
            movieNames[int(fields[0])] = fields[1]
    return movieNames


# In[43]:


def normalizeRating(x, mean):
    movie_id, rating = x
    norm_rating = rating - mean
    return (movie_id, norm_rating)


# In[44]:


def computeMovieMeanRating(ratings):
    counts_by_movieId = ratings.countByKey()
    sum_ratings = ratings.reduceByKey(lambda x, y: x+y)
    movie_avgs = sum_ratings.map(lambda x: (x[0], x[1]/counts_by_movieId[x[0]]))

    movie_avg_dict = movie_avgs.collectAsMap()
    return movie_avg_dict


# In[45]:


def filterDuplicates(row):
    (userID, ratings) = row
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return movie1 < movie2


# In[46]:


def makePairs(row):
    (user, ratings) = row
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return ((movie1, movie2), (rating1, rating2))


# In[47]:


def computeCosineSimilarity(ratingPairs):  
    numPairs = 0
    sum_xx = 0
    sum_yy = 0
    sum_xy = 0
    
    for row in ratingPairs:
        ratingX, ratingY = row
        ratingX = float(ratingX)
        ratingY = float(ratingY)
        sum_xx += float(ratingX) * float(ratingX)
        sum_yy += float(ratingY) * float(ratingY)
        sum_xy += float(ratingX) * float(ratingY)
        numPairs += 1

    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)

    score = 0
    if (denominator):
        score = (numerator / (float(denominator)))

    return (score, numPairs)


# In[48]:


movienames_dict = loadMovieNames()


# In[49]:


data = sc.textFile("ml-20m/ratings.csv")


# In[51]:


ratings = data.map(lambda l: l.split('\t')).map(lambda l: (int(l[0]), (int(l[1]), float(l[2]))))


# In[52]:


movie_mean_ratings = computeMovieMeanRating(ratings.values())
normalized_ratings = ratings.mapValues(lambda x: normalizeRating(x, movie_mean_ratings[x[0]]))
ratings = normalized_ratings


# In[53]:


ratings_user=ratings.map(lambda x: (x[1][0],(x[0],x[1][1])))


# In[54]:


user_mean_ratings = computeMovieMeanRating(ratings_user.values())


# In[55]:


ratingsPartitioned = ratings.partitionBy(100)


# In[56]:


joinedRatings = ratingsPartitioned.join(ratingsPartitioned)
uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)


# In[57]:


moviePairs = uniqueJoinedRatings.map(makePairs).partitionBy(100)


# In[58]:


moviePairRatings = moviePairs.groupByKey()


# In[59]:


moviePairSimilarities = moviePairRatings.mapValues(computeCosineSimilarity).persist()


# In[60]:


pair_dict=moviePairSimilarities.collectAsMap()


# In[ ]:


all_movies_mean = movie_mean_ratings.values().sum()/movie_mean_ratings.count()


# In[61]:





# In[62]:


users_rated_movies = ratings_user.map(lambda x : (x[0],x[1][0])).groupByKey()


# In[63]:


users_rated_movies_dict = users_rated_movies.map(lambda x : (x[0],list(x[1]))).collectAsMap()


# In[ ]:


def getpredictions(x):
    sim_sum = 0
    rating_sum = 0
    user_mean = 0
    for i in range(10):
        (userid,movieid,rating),(_,sm)=x[0]
        sim_sum += sm[0]
        rating_sum += rating
        user_mean += user_mean_ratings(userid)
    fraction = (sim_sum+rating_sum - user_mean)/sim_sum
    prediction = fraction + user_mean_ratings(userid)+ movie_mean_ratings[movieid] - all_movies_mean
    return ((userid,movieid,rating),prediction)


# In[64]:


_, test_data = data.randomSplit([0.05,0.95],seed = 0)


# In[65]:


test_data = test_data.map(lambda l: l.split('\t')).map(lambda l: (int(l[0]), (int(l[1]), float(l[2]))))


# In[66]:


joined1 = ratings.join(test_data)


# In[67]:


joined2 = joined1.cartesian(moviePairSimilarities)


# In[ ]:


joined2 = joined2.filter(lambda x : (x[0][1] in x[1][0][1]))


# In[ ]:


joined2 = joined2.map(lambda x : ((x[0][0],x[0][1],x[0][3][1]),(x[1][0],x[1][1]))     )


# In[ ]:


joined2 = joined2.groupByKey().sortBy(lambda x : -x[1][1][0])


# In[ ]:


joined2 = joined2.map(getpredictions)


# In[ ]:


prediction_cf = joined2.map(lambda x : (x[0][0],x[1][1],x[1][2],x[1]))
prediction_cols = ["userId","movieId","rating","prediction"]
prediction_cf = prediction_cf.toDF(prediction_cols)


# In[ ]:


evaluator = RegressionEvaluator(labelCol = "rating")
rmse_cf = evaluator.evaluate(prediction_cf)

mse_evaluator = RegressionEvaluator(metricName = "mse",labelCol = "rating")
mse_cf = mse_evaluator.evaluate(prediction_cf)

    # prediction.show()
evaluator = RegressionEvaluator(metricName = "r2",labelCol = "rating")
map_cf = evaluator.evaluate(prediction_cf)

print("printing metrics of cf model : ")
print(rmse_cf,mse_cf,map_cf)


# In[1]:


prediction_cf.write.mode('overwrite').csv("/user/sdindi/output/prediction_cf.csv")


# In[4]:


sc.stop()


# In[ ]:





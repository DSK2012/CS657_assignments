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

def main():
    spark = SparkSession.builder.appName('hw3').getOrCreate()

    data = spark.read.csv("ml-20m/ratings.csv",header=True)
    data1=spark.read.csv("ml-20m/movies.csv",header= True)

    data,_ = data.randomSplit([0.2,0.8],seed = 0)
    data1,_ = data1.randomSplit([0.2,0.8],seed = 0)

    # data.show()
    data = data.withColumnRenamed("rating","label")
    data = data.withColumn("userId",col("userId").cast(IntegerType()))\
                .withColumn("movieId",col("movieId").cast(IntegerType()))\
                .withColumn("label",col("label").cast(IntegerType()))
    data.show()
    data1=data1.withColumnRenamed("movieId", "m_movieId")

    final_data=data.join(data1,data.movieId ==  data1.m_movieId,"inner") 
    
    final_data=final_data.drop("m_movieId","timestamp")
    final_data.show(truncate=False)


    final_data = final_data.select('userId','movieId','rating',(lower(regexp_replace('title', "[^a-zA-Z0-9 ]", "")).alias('title')),(lower(regexp_replace('genres', "[^a-zA-Z0-9- ]", " ")).alias('genres')))

    #Tokenizing


    tokenizer = Tokenizer(inputCol="title", outputCol="title1")
    final_data = tokenizer.transform(final_data)

    tokenizer = Tokenizer(inputCol="genres", outputCol="genres1")
    final_data = tokenizer.transform(final_data)


    final_data=final_data.drop("title","genres")

    # final_data=final_data.withColumnRenamed("genres1", "genres")
    # final_data=final_data.withColumnRenamed("title1", "title")
    #Vocab_length

    cv = CountVectorizer()
    cv.setInputCol("title1")
    cv.setOutputCol("title2")
    model = cv.fit(final_data)
    title_vocab_length = len(model.vocabulary)

    cv = CountVectorizer()
    cv.setInputCol("genres1")
    cv.setOutputCol("genres2")
    model = cv.fit(final_data)
    genre_vocab_length = len(model.vocabulary)

    print(genre_vocab_length)

    print(title_vocab_length)

    final_data.show(truncate=False)

    #TF-idf

    hashingTF = HashingTF(inputCol="title1", outputCol="title3", numFeatures=title_vocab_length)
    featurizedData = hashingTF.transform(final_data)
   # alternatively, CountVectorizer can also be used to get term frequency vectors

    idf = IDF(inputCol="title3", outputCol="title")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

    hashingTF = HashingTF(inputCol="genres1", outputCol="genres3", numFeatures=genre_vocab_length)
    featurizedData = hashingTF.transform(rescaledData)
   # alternatively, CountVectorizer can also be used to get term frequency vectors

    idf = IDF(inputCol="genres3", outputCol="genres")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

    rescaledData=rescaledData.drop("title1","title3","genres1","genres3")


    rescaledData.show()
    rescaledData = rescaledData.withColumn("userId",col("userId").cast(IntegerType()))
    rescaledData = rescaledData.withColumn("movieId",col("movieId").cast(IntegerType()))
    rescaledData = rescaledData.withColumn("rating",col("rating").cast(IntegerType()))


    feature_columns = rescaledData.columns
    feature_columns.remove("rating")

    assembler = VectorAssembler(inputCols = feature_columns,outputCol = "features")
    rescaledData = assembler.transform(rescaledData)


   #  rescaledData.select("features",)
    rescaledData.show()

    (trainingData, testData) = rescaledData.randomSplit([0.8,0.2],seed = 0)
    print(trainingData.count())
    print(testData.count())
    # train_data, test_data = data.randomSplit([0.8,0.2],seed = 0)
    # print(train_data.count())
    # print(test_data.count())
    # train_data.cache()
    train_data = trainingData.select("userId","movieId","rating")
    test_data = testData.select("userId","movieId","rating")


    #---------------- imp ----------------
    # als = ALS(rank = 10, maxIter = 6,userCol = "userId", itemCol = "movieId",ratingCol = "rating",coldStartStrategy='drop')

    # model = als.fit(train_data)


    ##---------- cross validation -----------------
    als = ALS(userCol = "userId", itemCol = "movieId",ratingCol = "label",coldStartStrategy='drop')

    paramGrid = ParamGridBuilder()\
                .addGrid(als.maxIter,[10,15,20])\
                .addGrid(als.rank,[10,20,30])\
                .addGrid(als.regParam,[0.01,0.1])\
                .build()
    
    crossval = CrossValidator( estimator = als,
                                estimatorParamMaps = paramGrid,
                                evaluator = RegressionEvaluator(),
                                numFolds = 5)
    
    cvModel = crossval.fit(train_data)

    prediction_als = cvModel.transform(test_data)


    # prediction_als = model.transform(test_data).partitionBy(100)

    evaluator = RegressionEvaluator(labelCol = "rating")
    rmse_als = evaluator.evaluate(prediction_als)

    mse_evaluator = RegressionEvaluator(metricName = "mse",labelCol = "rating")
    mse_als = mse_evaluator.evaluate(prediction_als)

    # prediction.show()
    evaluator = RegressionEvaluator(metricName = "r2",labelCol = "rating")
    map_als = evaluator.evaluate(prediction_als)

    print("printing metrics of als model : ")
    print(rmse_als,mse_als,map_als)
    
    # predictions = model.recommendForUserSubset(pred_user,10)
    # print(predictions.collect())
    # predictions.show()
    # prediction.write.mode('overwrite').csv("/user/sdindi/output/prediction.csv")
    # with open("/user/sdindi/output/file.txt","w") as file:
    #     file.write("rmse of best model is : ")
    #     file.write(str(rmse))
    #     file.write("\nmse of best model is : ")
    #     file.write(str(mse))
    #     file.write("\nmap of best model is : ")
    #     file.write(str(map_))
    #     file.write("\n\n\n")
    #     file.write("best model obtained is : \n")
    #     file.write(cvModel.explainParams())
    
    # ---------------lr model ---------------
    lr = LinearRegression()
    lr.setLabelCol("rating")

    paramGrid = ParamGridBuilder() \
    .addGrid(lr.maxIter,[10,15,20])\
    .addGrid(lr.regParam, [0.1,0.001]) \
    .build()

    crossval = CrossValidator(estimator=lr,
                           estimatorParamMaps=paramGrid,
                           evaluator=RegressionEvaluator(labelCol = "rating"),
                           numFolds=5)

    cvModel = crossval.fit(trainingData)

    prediction_lr = cvModel.transform(testData)
    # prediction_lr.show()

    # evaluator = RegressionEvaluator(predictionCol="prediction",labelCol = "rating")
    # print(evaluator.evaluate(prediction_lr))
    evaluator = RegressionEvaluator(labelCol = "rating")
    rmse_lr = evaluator.evaluate(prediction_lr)

    mse_evaluator = RegressionEvaluator(metricName = "mse",labelCol = "rating")
    mse_lr = mse_evaluator.evaluate(prediction_lr)

    # prediction.show()
    evaluator = RegressionEvaluator(metricName = "r2",labelCol = "rating")
    map_lr = evaluator.evaluate(prediction_lr)

    prediction_cf = spark.read.csv("prediction_cf.csv")

    print("printing metrics of lr model : ")
    print(rmse_lr,mse_lr,map_lr)
    ##hybrid
    weight1 = 0.3
    weight2 = 0.7
    prediction1 = prediction_als
    prediction1 = prediction1.withColumnRenamed("prediction","prediction1")
    prediction2 = prediction_lr
    prediction2 = prediction2.withColumnRenamed("prediction","prediction2")
    prediction3 = prediction_cf
    prediction3 = prediction3.withColumnRenamed("prediction","prediction3")

    # prediction1.show()
    # prediction2.show()
    condition_list = [
        prediction1.userId == prediction2.userId,
        prediction1.movieId == prediction2.movieId,
        prediction1.label == prediction2.label
    ]

    condition_list_2 = [
        joined_df.userId == prediction3.userId,
        joined_df.movieId == prediction3.movieId,
        joined_df.label == prediction3.label
    ]

    joined_df = prediction1.join(prediction2, condition_list, "inner")
    joined_df.show()

    joined_df = joined_df.withColumn("p1", col("prediction1")*weight1)
    joined_df = joined_df.withColumn("p2", col("prediction2")*weight2)
    joined_df = joined_df.withColumn("final_prediction",col("p1")+col("p2"))

    joined_df.show()

    evaluator = RegressionEvaluator(labelCol = "rating",predictionCol = "final_prediction")
    rmse_hybrid = evaluator.evaluate(joined_df)

    mse_evaluator = RegressionEvaluator(metricName = "mse",labelCol = "rating",predictionCol = "final_prediction")
    mse_hybrid = mse_evaluator.evaluate(joined_df)

    # prediction.show()
    evaluator = RegressionEvaluator(metricName = "r2",labelCol = "rating",predictionCol = "final_prediction")
    map_hybrid = evaluator.evaluate(joined_df)

    print("printing metrics of hybrid : ")
    print(rmse_hybrid,mse_hybrid,map_hybrid)




    # ----------- hybrid with all three models -------------
    weight1 = 0.8
    weight3 = 0.2
    joined_df_new = joined_df.join(prediction2, condition_list, "inner")
    joined_df_new = joined_df_new.withColumn("p3", col("prediction2")*weight3)
    joined_df_new = joined_df_new.withColumn("p4", col("final_prediction")*weight1)
    joined_df_new = joined_df_new.withColumn("final_prediction_new",col("p3")+col("p4"))

    joined_df_new.show()

    evaluator = RegressionEvaluator(labelCol = "rating",predictionCol = "final_prediction_new")
    rmse_hybrid_three = evaluator.evaluate(joined_df_new)

    mse_evaluator = RegressionEvaluator(metricName = "mse",labelCol = "rating",predictionCol = "final_prediction_new")
    mse_hybrid_three = mse_evaluator.evaluate(joined_df_new)

    # prediction.show()
    evaluator = RegressionEvaluator(metricName = "r2",labelCol = "rating",predictionCol = "final_prediction_new")
    map_hybrid_three = evaluator.evaluate(joined_df_new)

    print("printing metrics of hybrid : ")
    print(rmse_hybrid_three,mse_hybrid_three,map_hybrid_three)

if __name__ == "__main__":
    main()
from __future__ import print_function
from cmath import nan
from os import truncate

from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col,isnan,when,count,lower,regexp_replace
from pyspark.ml.feature import Imputer,Tokenizer,StopWordsRemover,HashingTF, IDF, CountVectorizer, VectorAssembler
from pyspark.ml.classification import LogisticRegression,RandomForestClassifier,MultilayerPerceptronClassifier,LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator,BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
# from pyspark.pandas.DataFrame import na


if __name__ == "__main__":
    spark = SparkSession.builder.appName('hw2').getOrCreate()

    data = spark.read.option("header","true").csv("fake_job_postings.csv")
    
    
    #converting fraudulent column type from string to int
    data = data.withColumn("fraudulent",col("fraudulent").cast(IntegerType()))

    #removing invalid fradulent class
    data = data.filter((data.fraudulent == 0) | (data.fraudulent == 1))
    counts = data.select([count(when(col(c).isNull(), c)).alias(c) for c in data.columns] ).first()
   #  counts.show()
    print("!"*120,"\n")
    print("Count of number of missing values in each column")
    print(counts)


    print(data.count())
    count_percentage = data.count()/100

    cols_drop = [c for c in counts.asDict() if counts[c] > count_percentage]
    # print(cols_drop)
    data = data.drop(*cols_drop)

    sampled_data = data.sampleBy("fraudulent", fractions={0: 0.055, 1: 1}, seed=0)
    sampled_data.groupBy("fraudulent").count().show()

   #  sampled_data.show()

    sampled_data = sampled_data.select('job_id', (lower(regexp_replace('title', "[^a-zA-Z]", "")).alias('title')),(lower(regexp_replace('description', "[^a-zA-Z\\s]", "")).alias('description')),regexp_replace('telecommuting', "[^0-9]","nan").alias('telecommuting').cast(IntegerType()),regexp_replace('has_company_logo', "[^0-9]","nan").alias('has_company_logo').cast(IntegerType()),regexp_replace('has_questions', "[^0-9]","nan").alias('has_questions').cast(IntegerType()),'fraudulent')
    sampled_data = sampled_data.select('job_id', (lower(regexp_replace('title', "\s+", "")).alias('title')),(lower(regexp_replace('description', '/\s\s+/g', "")).alias('description')),regexp_replace('telecommuting', "[^0-9]","nan").alias('telecommuting').cast(IntegerType()),regexp_replace('has_company_logo', "[^0-9]","nan").alias('has_company_logo').cast(IntegerType()),regexp_replace('has_questions', "[^0-9]","nan").alias('has_questions').cast(IntegerType()),'fraudulent')
   #  sampled_data.show()
 # Imputing the columns that has null values with mean
    imputer = Imputer()
    imputer.setInputCols(["telecommuting", "has_company_logo","has_questions"])
    imputer.setOutputCols(["telecommuting", "has_company_logo","has_questions"])
    model = imputer.fit(sampled_data)
    sampled_data=model.transform(sampled_data)

    # Tokenize text columns

    tokenizer = Tokenizer(inputCol="title", outputCol="title1")
    sampled_data = tokenizer.transform(sampled_data)
    tokenizer = Tokenizer(inputCol="description", outputCol="description1")
    sampled_data= tokenizer.transform(sampled_data)
    #Remove stopwords from tokenized text
    remover = StopWordsRemover(inputCol="title1", outputCol="title11")
    sampled_data=remover.transform(sampled_data)
    remover = StopWordsRemover(inputCol="description1", outputCol="description11")
    sampled_data=remover.transform(sampled_data)
   #  sampled_data.show()
   #  sampled_data.groupby('fraudulent').count().show()


    #droping redundant columns
    cols_redundant = ["title","description","title1","description1"]
    sampled_data = sampled_data.drop(*cols_redundant)

    #renaming columns names
    sampled_data = sampled_data.withColumnRenamed("title11","title")\
                .withColumnRenamed("description11","description")
   #  sampled_data.show()

    cv = CountVectorizer()
    cv.setInputCol("description")
    cv.setOutputCol("des_words")
    model = cv.fit(sampled_data)
    description_vocab_length = len(model.vocabulary)

   #  cv = CountVectorizer()
    cv.setInputCol("title")
    cv.setOutputCol("title_words")
    model = cv.fit(sampled_data)
    title_vocab_length = len(model.vocabulary)
   #  sampled_data_rdd = sampled_data_rdd.flatMap()


    output_dict = {}
    
    #tf idf for description ---------
    hashingTF = HashingTF(inputCol="description", outputCol="drFeatures", numFeatures=1000)
    featurizedData = hashingTF.transform(sampled_data)
   # alternatively, CountVectorizer can also be used to get term frequency vectors

    idf = IDF(inputCol="drFeatures", outputCol="description_tfidf")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

   #  rescaledData.select("features",)
   #  rescaledData.show()
    
    hashingTF = HashingTF(inputCol="title", outputCol="titlefeatures", numFeatures=1000)
    featurizedData = hashingTF.transform(rescaledData)
   # alternatively, CountVectorizer can also be used to get term frequency vectors

    idf = IDF(inputCol="titlefeatures", outputCol="title_tfidf")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

   #  rescaledData.show()
    
    cols = ["title","description","drFeatures","titlefeatures","job_id"]
    rescaledData = rescaledData.drop(*cols)

   #  rescaledData.show()


    feature_columns = rescaledData.columns
    feature_columns.remove("fraudulent")


    assembler = VectorAssembler(inputCols = feature_columns,outputCol = "features")
    data_final = assembler.transform(rescaledData)

    data_final = data_final.withColumnRenamed("fraudulent","label")

    (trainingData, testData) = data_final.randomSplit([0.7,0.3],seed = 0)

    mcp = MultilayerPerceptronClassifier(seed=0)

    l = []
    paramGrid = ParamGridBuilder() \
    .addGrid(mcp.maxIter,[50])\
    .addGrid(mcp.layers, [[2003,1000,100,2]]) \
    .build()

    crossval = CrossValidator(estimator=mcp,
                           estimatorParamMaps=paramGrid,
                           evaluator=MulticlassClassificationEvaluator(),
                           numFolds=10)  # use 3+ folds in practice

   # # Run cross-validation, and choose the best set of parameters.
    cvModel = crossval.fit(trainingData)
    l.append(cvModel)


    prediction = cvModel.transform(testData)
   #  prediction.show()

    f1_evaluator = MulticlassClassificationEvaluator()
    l.append(f1_evaluator.evaluate(cvModel.transform(testData)))
    acc_evaluator = MulticlassClassificationEvaluator(metricName = "accuracy")
    l.append(acc_evaluator.evaluate(cvModel.transform(testData)))

    output_dict["MultilayerPerceptronClassifier"] = l




    hashingTF = HashingTF(inputCol="description", outputCol="drFeatures", numFeatures=description_vocab_length)
    featurizedData = hashingTF.transform(sampled_data)
   # alternatively, CountVectorizer can also be used to get term frequency vectors

    idf = IDF(inputCol="drFeatures", outputCol="description_tfidf")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

   #  rescaledData.select("features",)
   #  rescaledData.show()
    
    hashingTF = HashingTF(inputCol="title", outputCol="titlefeatures", numFeatures=title_vocab_length)
    featurizedData = hashingTF.transform(rescaledData)
   # alternatively, CountVectorizer can also be used to get term frequency vectors

    idf = IDF(inputCol="titlefeatures", outputCol="title_tfidf")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

   #  rescaledData.show()
    
    cols = ["title","description","drFeatures","titlefeatures","job_id"]
    rescaledData = rescaledData.drop(*cols)

   #  rescaledData.show()


    feature_columns = rescaledData.columns
    feature_columns.remove("fraudulent")


    assembler = VectorAssembler(inputCols = feature_columns,outputCol = "features")
    data_final = assembler.transform(rescaledData)
   #  data_final.show()
    data_final = data_final.withColumnRenamed("fraudulent","label")





    (trainingData, testData) = data_final.randomSplit([0.7,0.3],seed = 0)
    lr = LogisticRegression()
    lsvc = LinearSVC()
    rf = RandomForestClassifier(seed=0)

    lr_paramGrid = ParamGridBuilder() \
            .addGrid(lr.maxIter,[10])\
            .addGrid(lr.regParam, [0.2]) \
            .build()
    crossval = CrossValidator(estimator=lr,
                           estimatorParamMaps=lr_paramGrid,
                           evaluator=MulticlassClassificationEvaluator(),
                           numFolds=10)

    cvModel = crossval.fit(trainingData)

    predictions = cvModel.transform(testData)

    output_dict["LogisticRegression"] = [cvModel,f1_evaluator.evaluate(cvModel.transform(testData)),acc_evaluator.evaluate(cvModel.transform(testData))]




    lsvc_paramGrid = ParamGridBuilder() \
            .addGrid(lsvc.maxIter,[50])\
            .addGrid(lsvc.regParam, [0.4]) \
            .build()
    
    crossval = CrossValidator(estimator=lsvc,
                           estimatorParamMaps=lsvc_paramGrid,
                           evaluator=MulticlassClassificationEvaluator(),
                           numFolds=10)
   
    cvModel = crossval.fit(trainingData)

    predictions = cvModel.transform(testData)

    output_dict["LinearSVC"] = [cvModel,f1_evaluator.evaluate(cvModel.transform(testData)),acc_evaluator.evaluate(cvModel.transform(testData))]



    rf_paramGrid = ParamGridBuilder() \
            .addGrid(rf.numTrees,[80])\
            .addGrid(rf.maxDepth, [8]) \
            .build()
    
    crossval = CrossValidator(estimator=rf,
                           estimatorParamMaps=rf_paramGrid,
                           evaluator=MulticlassClassificationEvaluator(),
                           numFolds=10)
   
    cvModel = crossval.fit(trainingData)

    predictions = cvModel.transform(testData)

    output_dict["RandomForestClassifier"] = [cvModel,f1_evaluator.evaluate(cvModel.transform(testData)),acc_evaluator.evaluate(cvModel.transform(testData))]




















   #  def training_func(classifier,sampled_data,description_vocab_length,title_vocab_length):
   #    def FindMtype(classifier):
   #      # Intstantiate Model
   #      M = classifier
   #      # Learn what it is
   #      Mtype = type(M).__name__
        
   #      return Mtype
    
   #    Mtype = FindMtype(classifier)

   #    if Mtype in ("LogisticRegression","LinearSVC","RandomForestClassifier"):
   #       hashingTF = HashingTF(inputCol="description", outputCol="drFeatures", numFeatures=description_vocab_length)
   #       featurizedData = hashingTF.transform(sampled_data)
   # # alternatively, CountVectorizer can also be used to get term frequency vectors

   #       idf = IDF(inputCol="drFeatures", outputCol="description_tfidf")
   #       idfModel = idf.fit(featurizedData)
   #       rescaledData = idfModel.transform(featurizedData)

   #       hashingTF = HashingTF(inputCol="title", outputCol="titlefeatures", numFeatures=title_vocab_length)
   #       featurizedData = hashingTF.transform(rescaledData)
   # # alternatively, CountVectorizer can also be used to get term frequency vectors

   #       idf = IDF(inputCol="titlefeatures", outputCol="title_tfidf")
   #       idfModel = idf.fit(featurizedData)
   #       rescaledData = idfModel.transform(featurizedData)


   #    cols = ["title","description","drFeatures","titlefeatures","job_id"]
   #    rescaledData = rescaledData.drop(*cols)



   #    feature_columns = rescaledData.columns
   #    feature_columns.remove("fraudulent")


   #    assembler = VectorAssembler(inputCols = feature_columns,outputCol = "features")
   #    data_final = assembler.transform(rescaledData)
      

   #    (trainingData, testData) = data_final.randomSplit([0.7,0.3],seed = 0)

   #    lr = LogisticRegression()
   #    lsvc = LinearSVC()
   #    rf = RandomForestClassifier(seed=0)


   #    if Mtype in ("LogisticRegression"):
   #       paramGrid = ParamGridBuilder() \
   #          .addGrid(lr.maxIter,[10])\
   #          .addGrid(lr.regParam, [0.2]) \
   #          .build()
      
   #    if Mtype in ("LinearSVC"):
   #       paramGrid = ParamGridBuilder() \
   #          .addGrid(lsvc.maxIter,[50])\
   #          .addGrid(lsvc.regParam, [0.4]) \
   #          .build()
      
   #    if Mtype in ("RandomForestClassifier"):
   #       paramGrid = ParamGridBuilder() \
   #          .addGrid(rf.numTrees,[80])\
   #          .addGrid(rf.maxDepth, [8]) \
   #          .build()
      
      

   #    crossval = CrossValidator(estimator=classifier,
   #                         estimatorParamMaps=paramGrid,
   #                         evaluator=MulticlassClassificationEvaluator(),
   #                         numFolds=10)
   #    trainingData.show()
   #    cvModel = crossval.fit(trainingData)

   #    predictions = cvModel.transform(testData)

   #    #F1 calculation
   #    evaluator = MulticlassClassificationEvaluator()
   #    f1_score_test = evaluator.evaluate(predictions)
   #    #accuracy calculation
   #    accuracy_evaluator = MulticlassClassificationEvaluator(metricName = "accuracy")
   #    accuracy_test = accuracy_evaluator.evaluate(predictions)

   #    return cvModel, f1_score_test, accuracy_test,Mtype



   #  classifiers = [
   #                LogisticRegression(),
   #                LinearSVC(),
   #                RandomForestClassifier(),
   #                # MultilayerPerceptronClassifier()
   #  ]

   # #  output_dict = {}

   #  for classifier in classifiers:
   #    print("!"*100,classifier,"!"*100)
   #    best_model, f1_score, accuracy, model_name = training_func(classifier,sampled_data,description_vocab_length,title_vocab_length)
   #    output_dict[model_name] = [best_model,f1_score,accuracy]
    
   #  print(output_dict)
    
    for i in output_dict.keys():
      print("!"*50,i,"!"*50)
      print(output_dict[i][0].bestModel.explainParams())
      print(" F1 score on Test Data is : ",output_dict[i][1])
      print(" Accuracy obtained on Test Data is : ",output_dict[i][2])
      print("\n\n")


    spark.stop()
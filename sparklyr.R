#This script is just some training with apache via sparklyr interface 
#(which alow u to use dplyr and spark sql)

#Loading packages
library(sparklyr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(DBI)

#Connecting to spark env
sc <- spark_connect(master = "local")

#Loading more packages
library(c("nycflights13", "Lahman"))

#Loading data 
iris_tbl <- copy_to(sc, iris)
flights_tbl <- copy_to(sc, nycflights13::flights, "flights")
batting_tbl <- copy_to(sc, Lahman::Batting, "batting")

#returns the available datasets
src_tbls(sc)
#filtering
flights_tbl %>% filter(dep_delay == 2)

#Move IRIS data around
  #Copy data to Spark memory
import_iris <- copy_to(sc, iris, "spark_iris", overwrite = TRUE)
  #Creating train and test set
partition_iris <- sdf_partition( import_iris,training=0.5, testing=0.5)
sdf_register(partition_iris, c("spark_iris_training","spark_iris_test"))

#Creste reference to Spark Table
tidy_iris <- tbl(sc,"spark_iris_training") %>% select(Species, Petal_Length, Petal_Width)

#Spark ML decision tree
iris_dtree <-tidy_iris %>% ml_decision_tree(response="Species", features = c("Petal_Length", "Petal_Width"))

#Create ref to Spark Table
test_iris <- tbl(sc,"spark_iris_test")

#Predict
pred_iris <- sdf_predict(test_iris, iris_dtree) %>% collect
pred_iris

#Using SQL with DBI (https://github.com/r-dbi/DBI)

iris_preview <- dbGetQuery(sc, "select * from iris where Species != 'setosa' limit 15")
iris_preview


#Machine Learning functions within sparklyr
#We???ll use the built-in mtcars dataset, and see if we can predict a car???s fuel consumption
#(mpg) based on its weight (wt), and the number of cylinders the engine contains (cyl), assuming a linear rel.

#copy mtcars into spark

mtcars_tbl <- copy_to (sc, mtcars)
# filter our data set and then separate into train and test

partitions <- mtcars_tbl %>%
  filter(hp >= 100) %>%
  mutate(cyl8 = cyl == 8) %>%
  sdf_partition(training = 0.5, test = 0.5, seed = 1099)
partitions$training

#Summary of the ML model
fitmlcars <- partitions$training %>% ml_linear_regression(response = "mpg", features = c('wt', 'cyl'))
fitmlcars
summary(fitmlcars)


#Reading and writing files 
temp_csv <- tempfile(fileext = ".csv")
temp_parquet <- tempfile(fileext = ".parquet")
temp_json <- tempfile(fileext = ".json")


spark_write_csv(iris_tbl, temp_csv)
iris_csv_tbl <- spark_read_csv(sc, "iris_csv", temp_csv)
iris_csv_tbl

spark_disconnect(sc)

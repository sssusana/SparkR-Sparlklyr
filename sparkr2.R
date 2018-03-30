#based on the following tutorial: http://cfss.uchicago.edu/distrib003_spark.html#machine_learning_with_spark

#Installing and loading packages
#if (!require("stringi")) install.packages("stringi")
#spark_install(version = "2.1.0")
if(!require("sparklyr"))install.packages("sparklyr")
if (!require("dplyr")) install.packages("dplyr")
if (!require("titanic")) install.packages("titanic")
if (!require("purrr")) install.packages("purrr")

library(sparklyr)
library(dplyr)
library(titanic)
library(purrr)
library(ggplot2)
#library(rpart)

## alternative is to install spark like this
#1. spark_install(version = "1.6.0", hadoop_version = "2.6"
#2. sc <- spark_connect(master = "local", version = "1.6.0", hadoop_version = "2.6")

Sys.setenv(SPARK_HOME = "~/spark/spark-2.1.0-bin-hadoop2.7")
sc <- spark_connect(master = "local")

#Ok. Lets do something! 
titanic_tbl <-  copy_to(sc, titanic_train, "titanic", overwrite = TRUE)

##Features
#survival - Survival (0 = No; 1 = Yes)
#class - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
#sibsp - Number of Siblings/Spouses Aboard
#parch - Number of Parents/Children Aboard
#ticket - Ticket Number
#fare - Passenger Fare
#cabin - Cabin
#embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
#boat - Lifeboat (if survived)
#body - Body number (if did not survive and body was recovered)


#Feature eng with sparksql and dplyr
titanic2_tbl <- titanic_tbl %>%
  mutate(family_size = SibSp + Parch + 1) %>%
  mutate(Pclass = as.character(Pclass)) %>%
  filter(!is.na(Embarked)) %>%
  sdf_register("titanic2")

#(Should try to impute the age with decision tree)
#impute_age <- formula(Age ~ Survived + Pclass + Sex + SibSp +
#                     Parch + Fare + Embarked)

#Mini dic for dplyr-sql commands
#select ~ SELECT; filter ~ WHERE; arrange ~ ORDER; summarise ~ aggregators: sum, min, sd, etc.; mutate ~ operators: +, *, log, etc.
#mutate() adds new variables that are functions of existing variables


#Spark ML_transformations
titanic_final_tbl <- titanic2_tbl %>%
  mutate(family_size = as.numeric(family_size)) %>%
  sdf_mutate(
    family_sizes = ft_bucketizer(family_size, splits = c(1,2,5,12))
  ) %>%
  mutate(family_sizes = as.character(as.integer(family_sizes))) %>%
  sdf_register("titanic_final")

partition <- titanic_final_tbl %>% 
  mutate(Survived = as.numeric(Survived),
         SibSp = as.numeric(SibSp),
         Parch = as.numeric(Parch)) %>%
  select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, family_sizes) %>%
  sdf_partition(train = 0.75, test = 0.25, seed = 1234)

# Create table references
train_tbl <- partition$train
test_tbl <- partition$test

# Model survival as a function of several predictors
ml_formula <- formula(Survived ~ Pclass + Sex + Age + SibSp +
                        Parch + Fare + Embarked + family_sizes)

# Train a logistic regression model
(ml_log <- ml_logistic_regression(train_tbl, ml_formula))

# Decision Tree
ml_dt <- ml_decision_tree(train_tbl, ml_formula)

# Random Forest
ml_rf <- ml_random_forest(train_tbl, ml_formula)

# Gradient Boosted Tree
ml_gbt <- ml_gradient_boosted_trees(train_tbl, ml_formula)

# Naive Bayes
ml_nb <- ml_naive_bayes(train_tbl, ml_formula)

# Neural Network
ml_nn <- ml_multilayer_perceptron(train_tbl, ml_formula, layers = c(11, 15, 2))


ml_models <- list(
  "Logistic" = ml_log,
  "Decision Tree" = ml_dt,
  "Random Forest" = ml_rf,
  "Gradient Boosted Trees" = ml_gbt,
  "Naive Bayes" = ml_nb,
  "Neural Net" = ml_nn
)

# Create a function for scoring
score_test_data <- function(model, data = test_tbl){
  pred <- sdf_predict(model, data)
  select(pred, Survived, prediction)
}

# Score all the models
ml_score <- map(ml_models, score_test_data)
calculate_lift <- function(scored_data) {
  scored_data %>%
    mutate(bin = ntile(desc(prediction), 10)) %>% 
    group_by(bin) %>% 
    summarize(count = sum(Survived)) %>% 
    mutate(prop = count / sum(count)) %>% 
    arrange(bin) %>% 
    mutate(prop = cumsum(prop)) %>% 
    select(-count) %>% 
    collect() %>% 
    as.data.frame()
}

# Initialize results
ml_gains <- data_frame(
  bin = 1:10,
  prop = seq(0, 1, len = 10),
  model = "Base"
)

# Calculate lift
for(i in names(ml_score)){
  ml_gains <- ml_score[[i]] %>%
    calculate_lift %>%
    mutate(model = i) %>%
    bind_rows(ml_gains, .)
}

# Plot results
ggplot(ml_gains, aes(x = bin, y = prop, color = model)) +
  geom_point() +
  geom_line() +
  scale_color_brewer(type = "qual") +
  labs(title = "Lift Chart for Predicting Survival",
       subtitle = "Test Data Set",
       x = NULL,
       y = NULL)

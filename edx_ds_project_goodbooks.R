
###########################################
# Include libraries 
###########################################

library(tidyverse)
library(caret)
library(readr)
library(knitr)
library(Matrix)
library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)

install.packages("recommenderlab")
install.packages("recosystem")
library(recommenderlab)
library(recosystem)

####################################
# goodbooks-10k dataset
####################################

# goodbooks-10k dataset:
# https://www.kaggle.com/zygmunt/goodbooks-10k/
# There are 6 data files in the goodbooks-10k.zip file:
#   books.csv
#   book_tags.csv
#   tags.csv
#   ratings.csv
#   to_read.csv
#   sample_book.xml

# The separate data files can be downloaded from github repository
datafile_books <- "https://github.com/orientalpearls/edx-ds-cyo-goodbooks-10k/raw/master/books.csv"
datafile_book_tags <- "https://github.com/orientalpearls/edx-ds-cyo-goodbooks-10k/raw/master/book_tags.csv"
datafile_tags <- "https://github.com/orientalpearls/edx-ds-cyo-goodbooks-10k/raw/master/tags.csv"
datafile_ratings <- "https://github.com/orientalpearls/edx-ds-cyo-goodbooks-10k/raw/master/ratings.csv"

# Read the .csv data into R
books_all <- fread(datafile_books)
book_tags_all <- fread(datafile_book_tags)
tags_all <- fread(datafile_tags)
ratings_all <- fread(datafile_ratings)

# Data Cleaning

# Number of unique users and number of unique books
ratings_all %>% 
  summarize(num_books = n_distinct(book_id),
            num_users = n_distinct(user_id))

# Remove the duplicate ratings from the same user_id and book_id
ratings_n <- ratings_all %>% 
  group_by(user_id, book_id) %>% 
  mutate(N=n())
cat('Number of duplicate ratings: ', 
    nrow(filter(ratings_n, N > 1)))
ratings <- ratings_n %>% 
  filter(N == 1) %>%
  subset(select = -N)

# Remove users who rated fewer than 5 books
ratings_users_n <- ratings %>% 
  group_by(user_id) %>% 
  mutate(N_u = n())
cat('Number of users who rated fewer than 5 books: ', 
    uniqueN(filter(ratings_users_n, N_u <= 4)$user_id))
ratings <- ratings_users_n %>% 
  filter(N_u > 5) %>% 
  subset(select = -N_u)

# Data Exploration

# Distribution of ratings
ratings %>% 
  ggplot(aes(x = rating, fill = factor(rating))) +
  geom_bar(color = "grey20") + 
  guides(fill = FALSE)

# Number of ratings per user
ratings %>% 
  group_by(user_id) %>% 
  summarize(number_of_ratings_per_user = n()) %>% 
  ggplot(aes(number_of_ratings_per_user)) + 
  geom_bar(fill = "gray", color = "grey20") + 
  coord_cartesian(c(3, 50))

# Distribution of mean rating per user
ratings %>% 
  group_by(user_id) %>% 
  summarize(mean_rating_per_user = mean(rating)) %>% 
  ggplot(aes(mean_rating_per_user)) +
  geom_histogram(fill = "orange", color = "grey20")

# Plot of number of rated books per user vs. mean rating
tmp <- ratings %>% 
  group_by(user_id) %>% 
  summarize(mean_rating = mean(rating), number_of_rated_books_per_user = n())
tmp %>% 
  ggplot(aes(number_of_rated_books_per_user, mean_rating)) +
  geom_point() + 
  stat_smooth(method = "lm", color = "blue", size = 2, se = TRUE)

# Number of ratings per book
ratings %>% 
  group_by(book_id) %>% 
  summarize(number_of_ratings_per_book = n()) %>% 
  ggplot(aes(number_of_ratings_per_book)) + 
  geom_bar(fill = "gray", color = "grey20", width = 1) 

# Distribution of mean rating per book
ratings %>% 
  group_by(book_id) %>% 
  summarize(mean_rating_per_book = mean(rating)) %>% 
  ggplot(aes(mean_rating_per_book)) + 
  geom_histogram(fill = "orange", color = "grey20") + 
  coord_cartesian(c(1,5))

# Plot of number of ratings per book vs. mean rating
books_all %>% 
  filter(ratings_count < 1e+4) %>% 
  ggplot(aes(ratings_count, average_rating)) + 
  geom_point() + 
  stat_smooth(method = "lm", color = "blue", size = 2)

#####################################
# RMSE function definition
#####################################

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

################################################
# Method 1 Collaborative Filtering
# recommenderlab package
# RANDOM, POPULAR, UBCF
# It may take more than an hour to run the model,
# depending on how powerful the computer is. 
#################################################

##### Data Processing ####

#Ratings data first converted to matrix format
dimension_names <- list(user_id = sort(unique(ratings$user_id)), 
                        book_id = sort(unique(ratings$book_id)))
ratingmat <- spread(select(ratings, book_id, user_id, rating), book_id, rating) %>% 
  subset(select=-user_id)
ratingmat <- as.matrix(ratingmat)
dimnames(ratingmat) <- dimension_names

#Then converted to sparseMatrix format
tmpmat <- ratingmat
tmpmat[is.na(tmpmat)] <- 0
sparse_ratings <- as(tmpmat, "sparseMatrix")
rm(tmpmat)

#Finally converted into a recommenderlab realRatingMatrix format
real_ratings <- new("realRatingMatrix", data = sparse_ratings)
dim(real_ratings)

#### Exploring Parameters ####

# List of algorithms and default parameters
recommenderRegistry$get_entry_names()
recommenderRegistry$get_entries("RANDOM", dataType = "realRatingMatrix")
recommenderRegistry$get_entries("POPULAR", dataType = "realRatingMatrix")
recommenderRegistry$get_entries("UBCF", dataType = "realRatingMatrix")

# 10-fold cross validation
scheme <- evaluationScheme(real_ratings[1:500,], 
                           method = "cross-validation", 
                           k = 10, given = -1, goodRating = 5)

# Compare the performance for different algorithms and parameters
algorithms <- list("RANDOM" = list(name = "RANDOM", param = NULL),
                   "POPULAR" = list(name = "POPULAR", param = NULL),
                   "UBCF_05" = list(name = "UBCF", param = list(nn = 05)),
                   "UBCF_10" = list(name = "UBCF", param = list(nn = 10)),
                   "UBCF_30" = list(name = "UBCF", param = list(nn = 30)),
                   "UBCF_50" = list(name = "UBCF", param = list(nn = 50)))

# Evaluate the alogrithms with the given scheme            
results <- evaluate(scheme, algorithms, type = "ratings")
tmp <- lapply(results, function(x) slot(x, "results"))
res <- tmp %>% 
  lapply(function(x) unlist(lapply(x, function(x) unlist(x@cm[ ,"RMSE"])))) %>% 
  as.data.frame() %>% 
  gather(key = "Algorithm", value = "RMSE")
res %>% 
  ggplot(aes(Algorithm, RMSE, fill = Algorithm)) +
  geom_bar(stat="summary") + 
  geom_errorbar(stat="summary", width = 0.3, size = 0.8) +
  coord_cartesian(ylim = c(0.7, 1.2)) + 
  guides(fill = FALSE)

#### Creating training set (80% of data) and testing set ####
set.seed(1)
e <- evaluationScheme(real_ratings, method="split", train=0.80, given=-5)

#### RANDOM ####
# Predic ratings for RANDOM algorithm
model <- Recommender(getData(e, "train"), "RANDOM")
prediction <- predict(model, getData(e, "known"), type="ratings")
# Calculation of rmse for RANDOM method 
rmse_random <- calcPredictionAccuracy(prediction, getData(e, "unknown"))[1]
rmse_random

#### POPULAR ####
# Predict ratings for POPULAR algorithm
model <- Recommender(getData(e, "train"), method = "POPULAR", param=list(normalize = "center"))
prediction <- predict(model, getData(e, "known"), type="ratings")
#Calculation of rmse for POPULAR method
rmse_popular <- calcPredictionAccuracy(prediction, getData(e, "unknown"))[1]
rmse_popular

#### UBCF ####
# Predict ratings for UBCF using pearson similarity 
# and selected nn as 50 based on cross-validation
model <- Recommender(getData(e, "train"), method = "UBCF", 
                     param=list(normalize = "center", method="pearson", nn=50))
prediction <- predict(model, getData(e, "known"), type="ratings")
#Calculation of rmse for UBCF method 
rmse_ubcf <- calcPredictionAccuracy(prediction, getData(e, "unknown"))[1]
rmse_ubcf

###############################################
# Method 2 Matrix Factorization
# recosystem package
# It may take more than 10 minutes 
# to run the model
###############################################

##### Data Processing ####

#Creating traing and testing set (20% of the ratings dataset)
set.seed(1)
smp_size <- floor(0.20 * nrow(ratings))
in_train <- rep(TRUE, nrow(ratings))
in_train[sample(1:nrow(ratings), size = smp_size)] <- FALSE

train_set <- ratings[(in_train),]
test_set <- ratings[(!in_train),]

#Setup a test and train set for matrix factorization method for recosystem
train_set_mf <- data_memory(train_set$user_id, train_set$book_id, rating = train_set$rating)
test_set_mf <- data_memory(test_set$user_id, test_set$book_id, rating = NULL)

#### Tuning Parameters ####

# start a recosystem session
r = Reco()

# tune the hyper parameters
opts = r$tune(train_set_mf, opts = list(dim = c(1:30), lrate = c(0.1, 0.2),
                                        costp_l1 = 0, costq_l1 = 0,
                                        nthread = 1, niter = 10))
# optimal parameters
opts$min

#### Train the model and predict ratings ####

# train it using opts$min
r$train(train_set_mf, opts = opts$min)

# res$P and res$Q are the matrix P and Q respectively
res <- r$output(out_memory(), out_memory())

#Predicting ratings for test set
predicted_ratings <- r$predict(test_set_mf, out_memory())

#Calculation of rmse for matrix factorization method
rmse_mf <- RMSE(predicted_ratings, test_set$rating)
rmse_mf

##########################################
# Summary of RMSE for different algorithms
##########################################

rmse_results <- tibble(Method = "RANDOM", 
                       RMSE = rmse_random)
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="POPULAR",  
                                 RMSE = rmse_popular))
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="UBCF",  
                                 RMSE = rmse_ubcf))
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="MF",  
                                 RMSE = rmse_mf))
rmse_results %>% knitr::kable()


###########################################
# Generate .pdf file from r markdown (.Rmd)
# It may take one hour to run the models,
# depending how powerful the computer is.
###########################################

#install.packages("rmarkdown")
#library(rmarkdown)
#render('~/edx_ds_project_goodbooks.Rmd')








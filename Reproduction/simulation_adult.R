## Packages
library(dplyr)
library(Matrix)
library(glmnet)
library(mltools)
library(data.table)
library(grplasso)

## Data Input
### Train Dataset and cleaning
train <- read.csv("adult.csv", header = FALSE)
names(train) <- c(
  "age",
  "workclass",
  "fnlwgt",
  "education",
  "education-num",
  "marital-status",
  "occupation",
  "relationship",
  "race",
  "sex",
  "capital-gain",
  "capital-loss",
  "hours-per-week",
  "native-country",
  "income"
)
for (i in 1:dim(train)[1]) {
  for (j in 1:dim(train)[2]) {
    if (train[i, j] == ' ?') {
      train[i, j] <- NA
    }
  }
}
train <- na.omit(train)

### Test Dataset
test <- read.csv("adult_test.csv", header = FALSE)
names(test) <- c(
  "age",
  "workclass",
  "fnlwgt",
  "education",
  "education-num",
  "marital-status",
  "occupation",
  "relationship",
  "race",
  "sex",
  "capital-gain",
  "capital-loss",
  "hours-per-week",
  "native-country",
  "income"
)
for (i in 1:dim(test)[1]) {
  for (j in 1:dim(test)[2]) {
    if (test[i, j] == ' ?') {
      test[i, j] <- NA
    }
  }
}
test <- na.omit(test)

### One-hot encoding for first-order and second-order features
#### first-order features for training data
fof_train <- train[, (1:(dim(train)[2] - 1))] ##### First_Order_Feature of TRAIN dataset without one-hot encoding

for (i in 1:dim(fof_train)[2]) {
  if (typeof(fof_train[, i]) == "character") {
    fof_train[, i] <- as.factor(fof_train[, i])
  }
}

cols_index <- sapply(fof_train, class) != "factor"

fof_train[, cols_index] <-
  sapply(fof_train[, cols_index], scale)

fof_train_oh <- one_hot(data.table(fof_train)) ##### First_Order_Feature of TRAIN dataset with one-hot encoding

fof_train <- data.frame(fof_train)

fof_train_oh <- data.frame(fof_train_oh)

#### second-order features for training data
sof_train_oh <-
  data.frame(matrix(data = 0, nrow = dim(fof_train_oh)[1])) ##### Second_Order_Feature of TRAIN dataset without one-hot encoding

level <- rep(0, dim(fof_train)[2]) ##### Calculate the one hot encoding through first order feature

for (i in 1:dim(fof_train)[2]) {
  if (nlevels(fof_train[, i]) == 0) {
    level[i] <- 1
  } else{
    level[i] <- nlevels(fof_train[, i])
  }
} 

for (i in 2:length(level)) {
  level[i] <- level[i - 1] + level[i]
}

for (i in 1:level[length(level)]) {
  b <- which.min(abs(level-i))
  if(b != length(level)){
    pos_right <- level[b] + 1
    temp <-
      fof_train_oh[, c((pos_right):dim(fof_train_oh)[2])] * fof_train_oh[, i]
    sof_train_oh <- cbind(sof_train_oh, temp)
  }
}

sof_train_oh <- sof_train_oh[, -1]

feature_train_oh <- cbind(fof_train_oh, sof_train_oh)

#### first-order features for test data
fof_test <- test[, c(1:(dim(test)[2] - 1))] ##### First_Order_Feature of TEST dataset with one-hot encoding

for (i in 1:dim(fof_test)[2]) {
  if (typeof(fof_test[, i]) == "character") {
    fof_test[, i] <- as.factor(fof_test[, i])
  }
}

cols_index <- sapply(fof_test, class) != "factor"

fof_test[, cols_index] <-
  sapply(fof_test[, cols_index], scale)

fof_test_oh <- one_hot(data.table(fof_test))

fof_test <- data.frame(fof_test)

fof_test_oh <- data.frame(fof_test_oh)

#### second-order features for test data
sof_test_oh <-
  data.frame(matrix(data = 0, nrow = dim(fof_test_oh)[1])) ##### Second_Order_Feature of TEST dataset without one-hot encoding

level <- rep(0, dim(fof_test)[2]) ##### Calculate the one hot encoding through first order feature

for (i in 1:dim(fof_test)[2]) {
  if (nlevels(fof_test[, i]) == 0) {
    level[i] <- 1
  } else{
    level[i] <- nlevels(fof_test[, i])
  }
} 

for (i in 2:length(level)) {
  level[i] <- level[i - 1] + level[i]
}

for (i in 1:level[length(level)]) {
  b <- which.min(abs(level-i))
  if(b != length(level)){
    pos_right <- level[b] + 1
    temp <-
      fof_test_oh[, c((pos_right):dim(fof_test_oh)[2])] * fof_test_oh[, i]
    sof_test_oh <- cbind(sof_test_oh, temp)
  }
}

sof_test_oh <- sof_test_oh[, -1]

feature_test_oh <- cbind(fof_test_oh, sof_test_oh)


#### Setting the labels of outcome
outcome_train <- data.frame(train[, "income"])

for (i in 1:dim(outcome_train)[1]) {
  if (outcome_train[i, 1] == " <=50K") {
    outcome_train[i, 1] <- 0
    ##### Due to the equivalence of the Bayesian Linear Probit Model
  }
  if (outcome_train[i, 1] == " >50K") {
    outcome_train[i, 1] <- 1
  }
}

outcome_test <- data.frame(test[, "income"])

for (i in 1:dim(outcome_test)[1]) {
  if (outcome_test[i, 1] == " <=50K.") {
    outcome_test[i, 1] <- -1
  }
  if (outcome_test[i, 1] == " >50K.") {
    outcome_test[i, 1] <- 1
  }
}

outcome_train <- as.numeric(unlist(outcome_train))
outcome_test <- as.numeric(unlist(outcome_test))

## Model Setting
### Split the train set
day <- 6
size <- dim(train)[1] / day

### BLIP
#### Data pre-processing with adaptive LASSO
set.seed(1)

cv.ridge <-
  cv.glmnet(
    as.matrix(feature_train_oh),
    as.matrix(outcome_train),
    family = "binomial",
    gamma = 0,
    parallel = TRUE,
    type.measure = "class",
    nfolds = day
  )

weight <-
  1 / abs(matrix(coef(cv.ridge, s = "lambda.min")[, 1][2:(ncol(feature_train_oh) +
                                                            1)])) ^ 1

weight[weight[, 1] == Inf] <- 99999999

ada_lasso <-
  cv.glmnet(
    as.matrix(feature_train_oh),
    as.matrix(outcome_train),
    family = "binomial",
    gamma = 1,
    nfolds = day,
    standardize = TRUE,
    type.measure = "class",
    penalty.factor = weight,
    intercept = FALSE
  )

coef <- coef(ada_lasso, s = "lambda.1se")

index_coef <- which(coef != 0)

##### Searching for the whole variables

borders <- rep(0,(14 + 14*13/2))

level <- rep(0, dim(fof_test)[2]) 

for (i in 1:dim(fof_test)[2]) {
  if (nlevels(fof_test[, i]) == 0) {
    level[i] <- 1
  } else{
    level[i] <- nlevels(fof_test[, i])
  }
} 

l <- NULL

for(i in 1:(length(level)-1)){
  l <- c(l, level[i] * level[(i+1):length(level)])
}

for(i in 2:length(l)){
  l[i] <- l[i-1] + l[i]
}

borders <- c(level,level[length(level)]+l)

border_index <- rep(0,(14 + 14*13/2))

for(i in 1:length(index_coef)){
  border_index[which.min(abs(borders - index_coef[i]))] <- 1
}

l <- NULL

for(i in 1:length(borders)){
  if(border_index[i] == 1){
    l <- c(l,c((borders[i-1]+1) : borders[i]))
  }
}

selected_var <- row.names(coef)[l]

#### train the model with standard Gaussian prior & LogLoss on test set

select_x_train <- feature_train_oh[, selected_var]

beta_prior_mean <- rep(0, length(selected_var))

beta_prior_var <-
  diag(x = 1,
       nrow = length(selected_var),
       ncol = length(selected_var))

size <- dim(select_x_train)[1] / day

select_x_test <- feature_test_oh[, selected_var[-grep("Holand.Netherlands",selected_var)]]

outcome_temp_test <- outcome_test

for (i in 1:length(outcome_temp_test)) {
  if (outcome_temp_test[i] == -1) {
    outcome_temp_test[i] <- 0
  }
}

LogLoss <- rep(0, day)

for (r in 0:(day - 1)) {
  range <- c((size * r + 1):(size * (r + 1)))
  
  beta_post_var <-
    Matrix::solve(Matrix::solve(as.matrix(beta_prior_var)) + t(as.matrix(select_x_train[range, ])) %*% as.matrix(select_x_train[range, ]))
  
  beta_post_mean <-
    beta_post_var %*% (
      solve(as.matrix(beta_prior_var)) %*% as.matrix(beta_prior_mean) + t(as.matrix(select_x_train[range, ])) %*% outcome_train[range]
    )
  
  beta_prior_mean <- beta_post_mean
  
  beta_prior_var <- beta_post_var
  
  Pro <-
    pnorm(outcome_test * (as.matrix(select_x_test) %*% beta_post_mean[-grep("Holand.Netherlands",selected_var)]))
  
  LogLoss[r+1] <-
    sum(outcome_temp_test * log(Pro) + (1 - outcome_temp_test) * log(1 - Pro)) / (-length(outcome_temp_test))

}


### BLIPBayes
#### Get the EB estimator

LogLoss_BLIP_EB <- rep(0,day)

line <- dim(fof_test_oh)[2] ##### border of first and second features

n_first <- sum(l <= line)

n_second <- sum(l > line)

bt <- NULL

for(i in 1 : 20){
  range <- c(1 : size)
  
  bt <- c(bt,sample(x = range, size = length(range),replace = TRUE))
  
  beta_prior_mean <- rep(0, length(l))
  
  beta_prior_var <-
    diag(x = 1,
         nrow = length(l),
         ncol = length(l))
  
  beta_post_var <-
    Matrix::solve(Matrix::solve(as.matrix(beta_prior_var)) + t(as.matrix(select_x_train[bt, ])) %*% as.matrix(select_x_train[bt, ]))
  
  beta_post_mean <-
    beta_post_var %*% (
      solve(as.matrix(beta_prior_var)) %*% as.matrix(beta_prior_mean) + t(as.matrix(select_x_train[bt, ])) %*% outcome_train[bt]
    )
  
  var_first_EB <- mean((beta_post_mean[c(1:n_first)])^2 - diag(beta_post_var)[c(1:n_first)])
  
  var_second_EB <- mean((beta_post_mean[c(n_first:length(beta_post_mean))])^2 - diag(beta_post_var)[c(n_first:length(beta_post_mean))])
  
  if(var_first_EB > 0 && var_second_EB > 0){
    break
  }
}

beta_prior_mean <- rep(0,length(l))

d <- c(rep(var_first_EB,n_first),rep(var_second_EB,n_second))

beta_prior_var <- diag(d)

for(r in 0 : (day-1)){
  range <- c(1 : (size * r))
  
  beta_post_var <-
    Matrix::solve(Matrix::solve(as.matrix(beta_prior_var)) + t(as.matrix(select_x_train[range, ])) %*% as.matrix(select_x_train[range, ]))
  
  beta_post_mean <-
    beta_post_var %*% (
      solve(as.matrix(beta_prior_var)) %*% as.matrix(beta_prior_mean) + t(as.matrix(select_x_train[range, ])) %*% outcome_train[range]
    )
  
  
  beta_prior_mean <- beta_post_mean
  
  beta_prior_var <- beta_post_var
  
  Pro <-
    pnorm(outcome_test * (as.matrix(select_x_test) %*% beta_post_mean[-grep("Holand.Netherlands",selected_var)]))
  
  LogLoss_BLIP_EB[i] <-
    sum(outcome_temp_test * log(Pro) + (1 - outcome_temp_test) * log(1 - Pro)) / (-length(outcome_temp_test))
  
}



### BLIPTwice
#### Data pre-processing with adaptive LASSO

#### Get the EB estimator

#### re-train the model with new prior

#### LogLoss on test dataset

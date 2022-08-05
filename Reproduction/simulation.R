## Packages
library(dplyr)
library(Matrix)
library(glmnet)
library(mltools)
library(data.table)
library(grplasso)
library(sparklyr)
library(stringr)

## Functions Predefinition
upgrade_v <- function(t){
  result <- dnorm(t) / pnorm(t)
  return(result)
}

upgrade_w <- function(t){
  result <- upgrade_v(t) * (upgrade_v(t) + t)
  return(result)
}

MinMax <- function(x){
  x <- (x - min(x)) / (max(x) - min(x))
  return(x)
}

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
  sapply(fof_train[, cols_index], MinMax)

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
  sapply(fof_test[, cols_index], MinMax)

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

boots_index <- sample(c(1:size),(8*size),replace = TRUE)
boots_data <- feature_train_oh[boots_index,]
boots_outcome <- outcome_train[boots_index]

cv.ridge <-
  cv.glmnet(
    as.matrix(boots_data),
    as.matrix(boots_outcome),
    family = "binomial",
    gamma = 0,
    type.measure = "class",
    nfolds = 10
  )

weight <-
  1 / abs(matrix(coef(cv.ridge, s = "lambda.min")[, 1][2:(ncol(boots_data) +
                                                            1)])) ^ 1

weight[weight[, 1] == Inf] <- 99999999

ada_lasso <-
  cv.glmnet(
    as.matrix(boots_data),
    as.matrix(boots_outcome),
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

index_coef <- sort(union(index_coef,c(1:dim(fof_train_oh)[2])))

##### Searching for the whole variables

selected_var <- row.names(coef)[(index_coef)]

selected_var <- intersect(selected_var,colnames(feature_test_oh))

#### train the model with standard Gaussian prior & LogLoss on test set

select_x_train <- feature_train_oh[, selected_var]

beta_prior_mean <- data.frame(t(rep(0, length(selected_var))))

beta_prior_var <-
  data.frame(t(rep(1,length(selected_var))))

size <- dim(select_x_train)[1] / day

select_x_test <- feature_test_oh[, selected_var]

outcome_temp_test <- outcome_test

for (i in 1:length(outcome_temp_test)) {
  if (outcome_temp_test[i] == -1) {
    outcome_temp_test[i] <- 0
  }
}


LogLoss <- rep(0, day) 

beta_prior_mean <- as.matrix(rep(0, length(selected_var)))

beta_prior_var <- diag(1,length(selected_var))

set.seed(1)

for(i in 1:day){
  range <- c((size*(i-1)+1):(size*i))
  y_star <- as.matrix(select_x_train[range,]) %*% beta_prior_mean + rnorm(size)
  beta_post_var <- base::solve(base::solve(beta_prior_var) + t(select_x_train[range,]) %*% as.matrix(select_x_train[range,]))
  beta_post_mean <- beta_post_var %*% (base::solve(beta_prior_var) %*% beta_prior_mean + t(select_x_train[range,]) %*%  y_star)
  beta_prior_mean <- beta_post_mean
  beta_prior_var <- beta_post_var
  
  Pro <-
    pnorm(outcome_test * ((as.matrix(select_x_test) %*% (beta_post_mean))))
  
  LogLoss[i] <-
    sum(outcome_temp_test * log(Pro) + (1 - outcome_temp_test) * log(1 - Pro)) / (-length(outcome_temp_test))
}

### BLIPBayes
#### Get the EB estimator

LogLoss_BLIP_EB <- rep(0,day)

line <- dim(fof_test_oh)[2] ##### border of first and second features

n_first <- length(intersect(selected_var,colnames(fof_train_oh)))

n_second <- length(selected_var) - n_first

bt <- NULL

for(i in 1 : 8){
  range <- c(1 : size)
 
  bt <- c(bt,sample(x = range, size = length(range),replace = TRUE))

}   ##### Bootstrap

##### Compute the meta-prior variance of two categories

beta_prior_mean <- as.matrix(rep(0, length(selected_var)))

beta_prior_var <- diag(1,length(selected_var))

for(i in 1:8){
  range <- bt[(size*(i-1)+1):(size*i)]
  y_star <- as.matrix(select_x_train[range,]) %*% beta_prior_mean + rnorm(size)
  beta_post_var <- base::solve(base::solve(beta_prior_var) + t(select_x_train[range,]) %*% as.matrix(select_x_train[range,]))
  beta_post_mean <- beta_post_var %*% (base::solve(beta_prior_var) %*% beta_prior_mean + t(select_x_train[range,]) %*%  y_star)
  beta_prior_mean <- beta_post_mean
  beta_prior_var <- beta_post_var
}

var_first_EB <- sum((beta_post_mean[c(1:n_first)])^2 - beta_post_var[c(1:n_first)]) / n_first

var_second_EB <- sum((beta_post_mean[c(n_first:length(beta_post_mean))])^2 - beta_post_var[c(n_first:length(beta_post_mean))]) / n_second

##### Restart the BLIP model with new meta-prior

set.seed(1)

beta_prior_mean <- as.matrix(rep(0, length(selected_var)))

beta_prior_var <- as.matrix(diag(c(rep(var_first_EB,n_first),rep(var_second_EB,n_second))))

for(i in 1:day){
  range <- c((size*(i-1)+1):(size*i))
  y_star <- as.matrix(select_x_train[range,]) %*% beta_prior_mean + rnorm(size)
  beta_post_var <- base::solve(base::solve(beta_prior_var) + t(select_x_train[range,]) %*% as.matrix(select_x_train[range,]))
  beta_post_mean <- beta_post_var %*% (base::solve(beta_prior_var) %*% beta_prior_mean + t(select_x_train[range,]) %*%  y_star)
  beta_prior_mean <- beta_post_mean
  beta_prior_var <- beta_post_var
  
  Pro <-
    pnorm(outcome_test * ((as.matrix(select_x_test) %*% (beta_post_mean))))
  
  LogLoss_BLIP_EB[i] <-
    sum(outcome_temp_test * log(Pro) + (1 - outcome_temp_test) * log(1 - Pro)) / (-length(outcome_temp_test))
}


### BLIPTwice
LogLoss_BLIP_EB_twice <- rep(0,day)

line <- dim(fof_test_oh)[2] ##### border of first and second features

n_first <- length(intersect(selected_var,colnames(fof_train_oh)))

n_second <- length(selected_var) - n_first

bt <- NULL

for(i in 1 : 8){
  range <- c(1 : size)
  
  bt <- c(bt,sample(x = range, size = length(range),replace = TRUE))
  
}

bt <- c(bt,c(1:size))

beta_prior_mean <- as.matrix(rep(0, length(selected_var)))

beta_prior_var <- diag(1,length(selected_var))

for(i in 1:9){
  range <- bt[(size*(i-1)+1):(size*i)]
  y_star <- as.matrix(select_x_train[range,]) %*% beta_prior_mean + rnorm(size)
  beta_post_var <- base::solve(base::solve(beta_prior_var) + t(select_x_train[range,]) %*% as.matrix(select_x_train[range,]))
  beta_post_mean <- beta_post_var %*% (base::solve(beta_prior_var) %*% beta_prior_mean + t(select_x_train[range,]) %*%  y_star)
  beta_prior_mean <- beta_post_mean
  beta_prior_var <- beta_post_var
}

var_first_EB <- sum((beta_post_mean[c(1:n_first)])^2 - beta_post_var[c(1:n_first)]) / n_first

var_second_EB <- sum((beta_post_mean[c(n_first:length(beta_post_mean))])^2 - beta_post_var[c(n_first:length(beta_post_mean))]) / n_second

set.seed(1)

beta_prior_mean <- as.matrix(rep(0, length(selected_var)))

beta_prior_var <- as.matrix(diag(c(rep(var_first_EB,n_first),rep(var_second_EB,n_second))))

for(i in 1:day){
  range <- c((size*(i-1)+1):(size*i))
  y_star <- as.matrix(select_x_train[range,]) %*% beta_prior_mean + rnorm(size)
  beta_post_var <- base::solve(base::solve(beta_prior_var) + t(select_x_train[range,]) %*% as.matrix(select_x_train[range,]))
  beta_post_mean <- beta_post_var %*% (base::solve(beta_prior_var) %*% beta_prior_mean + t(select_x_train[range,]) %*%  y_star)
  beta_prior_mean <- beta_post_mean
  beta_prior_var <- beta_post_var
  
  Pro <-
    pnorm(outcome_test * ((as.matrix(select_x_test) %*% (beta_post_mean))))
  
  LogLoss_BLIP_EB_twice[i] <-
    sum(outcome_temp_test * log(Pro) + (1 - outcome_temp_test) * log(1 - Pro)) / (-length(outcome_temp_test))
}


#### Plots for three models
plot(LogLoss,ylim = c(0.7,0.72),type = "b",ylab = "Logloss", xlab = "day", col = "green", main = "All 1st order features")
lines(LogLoss_BLIP_EB,type = "b",col = "blue")
lines(LogLoss_BLIP_EB_twice, type = "b",col = "red")
legend(x = 'topleft', 
       legend = c(expression(BLIP),expression(BLIPBayes),expression(BLIPTwice)),
       col = c('green','blue','red'),
       lty = 1)

### BLIPBayes with reset time t = 3

LogLoss_BLIP_EB_r3 <- rep(0,day)

line <- dim(fof_test_oh)[2] ##### border of first and second features

n_first <- length(intersect(selected_var,colnames(fof_train_oh)))

n_second <- length(selected_var) - n_first

bt <- NULL

for(i in 1 : 8){
  range <- c(1 : (3 * size))
  
  bt <- c(bt,sample(x = range, size = length(range),replace = TRUE))
  
}   ##### Bootstrap

##### Compute the meta-prior variance of two categories

beta_prior_mean <- as.matrix(rep(0, length(selected_var)))

beta_prior_var <- diag(1,length(selected_var))

for(i in 1:8){
  range <- bt[(size*(i-1)+1):(size*i)]
  y_star <- as.matrix(select_x_train[range,]) %*% beta_prior_mean + rnorm(size)
  beta_post_var <- base::solve(base::solve(beta_prior_var) + t(select_x_train[range,]) %*% as.matrix(select_x_train[range,]))
  beta_post_mean <- beta_post_var %*% (base::solve(beta_prior_var) %*% beta_prior_mean + t(select_x_train[range,]) %*%  y_star)
  beta_prior_mean <- beta_post_mean
  beta_prior_var <- beta_post_var
}

var_first_EB <- sum((beta_post_mean[c(1:n_first)])^2 - beta_post_var[c(1:n_first)]) / n_first

var_second_EB <- sum((beta_post_mean[c(n_first:length(beta_post_mean))])^2 - beta_post_var[c(n_first:length(beta_post_mean))]) / n_second

##### Restart the BLIP model with new meta-prior

set.seed(1)

beta_prior_mean <- as.matrix(rep(0, length(selected_var)))

beta_prior_var <- as.matrix(diag(c(rep(var_first_EB,n_first),rep(var_second_EB,n_second))))

for(i in 1:day){
  range <- c((size*(i-1)+1):(size*i))
  y_star <- as.matrix(select_x_train[range,]) %*% beta_prior_mean + rnorm(size)
  beta_post_var <- base::solve(base::solve(beta_prior_var) + t(select_x_train[range,]) %*% as.matrix(select_x_train[range,]))
  beta_post_mean <- beta_post_var %*% (base::solve(beta_prior_var) %*% beta_prior_mean + t(select_x_train[range,]) %*%  y_star)
  beta_prior_mean <- beta_post_mean
  beta_prior_var <- beta_post_var
  
  Pro <-
    pnorm(outcome_test * ((as.matrix(select_x_test) %*% (beta_post_mean))))
  
  LogLoss_BLIP_EB_r3[i] <-
    sum(outcome_temp_test * log(Pro) + (1 - outcome_temp_test) * log(1 - Pro)) / (-length(outcome_temp_test))
}

plot(LogLoss,ylim = c(0.7,0.72),type = "b",ylab = "Logloss", xlab = "day",col = "green",main = "Effect of Prior Reset Time")
lines(LogLoss_BLIP_EB,type = "b",col = "blue")
lines(LogLoss_BLIP_EB_r3, type = "b",col = "red")
legend(x = 'topleft', 
       legend = c(expression(BLIP),expression(BLIPBayes),expression(ResetTime_3)),
       col = c('green','blue','red'),
       lty = 1)

#### Effect of Variance
##### Variance = 5
var_first_EB <- 5

var_second_EB <- 5

##### Restart the BLIP model with new meta-prior

set.seed(1)

LogLoss_BLIP_EB_v5 <- rep(0,day)

beta_prior_mean <- as.matrix(rep(0, length(selected_var)))

beta_prior_var <- as.matrix(diag(c(rep(var_first_EB,n_first),rep(var_second_EB,n_second))))

for(i in 1:day){
  range <- c((size*(i-1)+1):(size*i))
  y_star <- as.matrix(select_x_train[range,]) %*% beta_prior_mean + rnorm(size)
  beta_post_var <- base::solve(base::solve(beta_prior_var) + t(select_x_train[range,]) %*% as.matrix(select_x_train[range,]))
  beta_post_mean <- beta_post_var %*% (base::solve(beta_prior_var) %*% beta_prior_mean + t(select_x_train[range,]) %*%  y_star)
  beta_prior_mean <- beta_post_mean
  beta_prior_var <- beta_post_var
  
  Pro <-
    pnorm(outcome_test * ((as.matrix(select_x_test) %*% (beta_post_mean))))
  
  LogLoss_BLIP_EB_v5[i] <-
    sum(outcome_temp_test * log(Pro) + (1 - outcome_temp_test) * log(1 - Pro)) / (-length(outcome_temp_test))
}

##### Variance = 0.1
var_first_EB <- 0.1

var_second_EB <- 0.1

##### Restart the BLIP model with new meta-prior

set.seed(1)

LogLoss_BLIP_EB_v01 <- rep(0,day)

beta_prior_mean <- as.matrix(rep(0, length(selected_var)))

beta_prior_var <- as.matrix(diag(c(rep(var_first_EB,n_first),rep(var_second_EB,n_second))))

for(i in 1:day){
  range <- c((size*(i-1)+1):(size*i))
  y_star <- as.matrix(select_x_train[range,]) %*% beta_prior_mean + rnorm(size)
  beta_post_var <- base::solve(base::solve(beta_prior_var) + t(select_x_train[range,]) %*% as.matrix(select_x_train[range,]))
  beta_post_mean <- beta_post_var %*% (base::solve(beta_prior_var) %*% beta_prior_mean + t(select_x_train[range,]) %*%  y_star)
  beta_prior_mean <- beta_post_mean
  beta_prior_var <- beta_post_var
  
  Pro <-
    pnorm(outcome_test * ((as.matrix(select_x_test) %*% (beta_post_mean))))
  
  LogLoss_BLIP_EB_v01[i] <-
    sum(outcome_temp_test * log(Pro) + (1 - outcome_temp_test) * log(1 - Pro)) / (-length(outcome_temp_test))
}

##### Variance = 0.01 
var_first_EB <- 0.01

var_second_EB <- 0.01

##### Restart the BLIP model with new meta-prior

set.seed(1)

LogLoss_BLIP_EB_v001 <- rep(0,day)

beta_prior_mean <- as.matrix(rep(0, length(selected_var)))

beta_prior_var <- as.matrix(diag(c(rep(var_first_EB,n_first),rep(var_second_EB,n_second))))

for(i in 1:day){
  range <- c((size*(i-1)+1):(size*i))
  y_star <- as.matrix(select_x_train[range,]) %*% beta_prior_mean + rnorm(size)
  beta_post_var <- base::solve(base::solve(beta_prior_var) + t(select_x_train[range,]) %*% as.matrix(select_x_train[range,]))
  beta_post_mean <- beta_post_var %*% (base::solve(beta_prior_var) %*% beta_prior_mean + t(select_x_train[range,]) %*%  y_star)
  beta_prior_mean <- beta_post_mean
  beta_prior_var <- beta_post_var
  
  Pro <-
    pnorm(outcome_test * ((as.matrix(select_x_test) %*% (beta_post_mean))))
  
  LogLoss_BLIP_EB_v001[i] <-
    sum(outcome_temp_test * log(Pro) + (1 - outcome_temp_test) * log(1 - Pro)) / (-length(outcome_temp_test))
}

plot(LogLoss,ylim = c(0.7,0.72),type = "b",ylab = "Logloss", col = "green", main = "Effect of Prior Variance", xlab = "day")
lines(LogLoss_BLIP_EB,type = "b",col = "blue")
lines(LogLoss_BLIP_EB_v5, type = "b",col = "red")
lines(LogLoss_BLIP_EB_v01, type = "b",col = "black")
lines(LogLoss_BLIP_EB_v001, type = "b",col = "grey")
legend(x = 'topleft', 
       legend = c(expression(BLIP),expression(BLIPBayes),expression(5),expression(0.1),expression(0.01)),
       col = c('green','blue','red','black','grey'),
       lty = 1,
       xpd = TRUE,
       inset = -0.1)


### First-order features
index_coef <- which(coef != 0)

##### Searching for the whole variables

selected_var <- row.names(coef)[(index_coef)]

selected_var <- intersect(selected_var,colnames(feature_test_oh))

#### train the model with standard Gaussian prior & LogLoss on test set

select_x_train <- feature_train_oh[, selected_var]

beta_prior_mean <- data.frame(t(rep(0, length(selected_var))))

beta_prior_var <-
  data.frame(t(rep(1,length(selected_var))))

size <- dim(select_x_train)[1] / day

select_x_test <- feature_test_oh[, selected_var]

outcome_temp_test <- outcome_test

for (i in 1:length(outcome_temp_test)) {
  if (outcome_temp_test[i] == -1) {
    outcome_temp_test[i] <- 0
  }
}


LogLoss <- rep(0, day)

beta_prior_mean <- as.matrix(rep(0, length(selected_var)))

beta_prior_var <- diag(1,length(selected_var))

set.seed(1)

for(i in 1:day){
  range <- c((size*(i-1)+1):(size*i))
  y_star <- as.matrix(select_x_train[range,]) %*% beta_prior_mean + rnorm(size)
  beta_post_var <- base::solve(base::solve(beta_prior_var) + t(select_x_train[range,]) %*% as.matrix(select_x_train[range,]))
  beta_post_mean <- beta_post_var %*% (base::solve(beta_prior_var) %*% beta_prior_mean + t(select_x_train[range,]) %*%  y_star)
  beta_prior_mean <- beta_post_mean
  beta_prior_var <- beta_post_var
  
  Pro <-
    pnorm(outcome_test * ((as.matrix(select_x_test) %*% (beta_post_mean))))
  
  LogLoss[i] <-
    sum(outcome_temp_test * log(Pro) + (1 - outcome_temp_test) * log(1 - Pro)) / (-length(outcome_temp_test))
}

### BLIPBayes
#### Get the EB estimator

LogLoss_BLIP_EB <- rep(0,day)

line <- dim(fof_test_oh)[2] ##### border of first and second features

n_first <- length(intersect(selected_var,colnames(fof_train_oh)))

n_second <- length(selected_var) - n_first

bt <- NULL

for(i in 1 : 8){
  range <- c(1 : size)
  
  bt <- c(bt,sample(x = range, size = length(range),replace = TRUE))
  
}   ##### Bootstrap

##### Compute the meta-prior variance of two categories

beta_prior_mean <- as.matrix(rep(0, length(selected_var)))

beta_prior_var <- diag(1,length(selected_var))

for(i in 1:8){
  range <- bt[(size*(i-1)+1):(size*i)]
  y_star <- as.matrix(select_x_train[range,]) %*% beta_prior_mean + rnorm(size)
  beta_post_var <- base::solve(base::solve(beta_prior_var) + t(select_x_train[range,]) %*% as.matrix(select_x_train[range,]))
  beta_post_mean <- beta_post_var %*% (base::solve(beta_prior_var) %*% beta_prior_mean + t(select_x_train[range,]) %*%  y_star)
  beta_prior_mean <- beta_post_mean
  beta_prior_var <- beta_post_var
}

var_first_EB <- sum((beta_post_mean[c(1:n_first)])^2 - beta_post_var[c(1:n_first)]) / n_first

var_second_EB <- sum((beta_post_mean[c(n_first:length(beta_post_mean))])^2 - beta_post_var[c(n_first:length(beta_post_mean))]) / n_second

##### Restart the BLIP model with new meta-prior

set.seed(1)

beta_prior_mean <- as.matrix(rep(0, length(selected_var)))

beta_prior_var <- as.matrix(diag(c(rep(var_first_EB,n_first),rep(var_second_EB,n_second))))

for(i in 1:day){
  range <- c((size*(i-1)+1):(size*i))
  y_star <- as.matrix(select_x_train[range,]) %*% beta_prior_mean + rnorm(size)
  beta_post_var <- base::solve(base::solve(beta_prior_var) + t(select_x_train[range,]) %*% as.matrix(select_x_train[range,]))
  beta_post_mean <- beta_post_var %*% (base::solve(beta_prior_var) %*% beta_prior_mean + t(select_x_train[range,]) %*%  y_star)
  beta_prior_mean <- beta_post_mean
  beta_prior_var <- beta_post_var
  
  Pro <-
    pnorm(outcome_test * ((as.matrix(select_x_test) %*% (beta_post_mean))))
  
  LogLoss_BLIP_EB[i] <-
    sum(outcome_temp_test * log(Pro) + (1 - outcome_temp_test) * log(1 - Pro)) / (-length(outcome_temp_test))
}


### BLIPTwice
LogLoss_BLIP_EB_twice <- rep(0,day)

line <- dim(fof_test_oh)[2] ##### border of first and second features

n_first <- length(intersect(selected_var,colnames(fof_train_oh)))

n_second <- length(selected_var) - n_first

bt <- NULL

for(i in 1 : 8){
  range <- c(1 : size)
  
  bt <- c(bt,sample(x = range, size = length(range),replace = TRUE))
  
}

bt <- c(bt,c(1:size))

beta_prior_mean <- as.matrix(rep(0, length(selected_var)))

beta_prior_var <- diag(1,length(selected_var))

for(i in 1:9){
  range <- bt[(size*(i-1)+1):(size*i)]
  y_star <- as.matrix(select_x_train[range,]) %*% beta_prior_mean + rnorm(size)
  beta_post_var <- base::solve(base::solve(beta_prior_var) + t(select_x_train[range,]) %*% as.matrix(select_x_train[range,]))
  beta_post_mean <- beta_post_var %*% (base::solve(beta_prior_var) %*% beta_prior_mean + t(select_x_train[range,]) %*%  y_star)
  beta_prior_mean <- beta_post_mean
  beta_prior_var <- beta_post_var
}

var_first_EB <- sum((beta_post_mean[c(1:n_first)])^2 - beta_post_var[c(1:n_first)]) / n_first

var_second_EB <- sum((beta_post_mean[c(n_first:length(beta_post_mean))])^2 - beta_post_var[c(n_first:length(beta_post_mean))]) / n_second

set.seed(1)

beta_prior_mean <- as.matrix(rep(0, length(selected_var)))

beta_prior_var <- as.matrix(diag(c(rep(var_first_EB,n_first),rep(var_second_EB,n_second))))

for(i in 1:day){
  range <- c((size*(i-1)+1):(size*i))
  y_star <- as.matrix(select_x_train[range,]) %*% beta_prior_mean + rnorm(size)
  beta_post_var <- base::solve(base::solve(beta_prior_var) + t(select_x_train[range,]) %*% as.matrix(select_x_train[range,]))
  beta_post_mean <- beta_post_var %*% (base::solve(beta_prior_var) %*% beta_prior_mean + t(select_x_train[range,]) %*%  y_star)
  beta_prior_mean <- beta_post_mean
  beta_prior_var <- beta_post_var
  
  Pro <-
    pnorm(outcome_test * ((as.matrix(select_x_test) %*% (beta_post_mean))))
  
  LogLoss_BLIP_EB_twice[i] <-
    sum(outcome_temp_test * log(Pro) + (1 - outcome_temp_test) * log(1 - Pro)) / (-length(outcome_temp_test))
}

plot(LogLoss,ylim = c(0.7,0.72),type = "b",ylab = "Logloss", xlab = "day", col = "green", main = "Select 1st features")
lines(LogLoss_BLIP_EB,type = "b",col = "blue")
lines(LogLoss_BLIP_EB_twice, type = "b",col = "red")
legend(x = 'topleft', 
       legend = c(expression(BLIP),expression(BLIPBayes),expression(BLIPTwice)),
       col = c('green','blue','red'),
       lty = 1)

### Small batches
### Split the train set

day <- 30
size <- 1000

### BLIP
#### Data pre-processing with adaptive LASSO
set.seed(1)

boots_index <- sample(c(1:size),(12*size),replace = TRUE)
boots_data <- feature_train_oh[boots_index,]
boots_outcome <- outcome_train[boots_index]

cv.ridge <-
  cv.glmnet(
    as.matrix(boots_data),
    as.matrix(boots_outcome),
    family = "binomial",
    gamma = 0,
    type.measure = "class",
    nfolds = 10
  )

weight <-
  1 / abs(matrix(coef(cv.ridge, s = "lambda.min")[, 1][2:(ncol(boots_data) +
                                                            1)])) ^ 1

weight[weight[, 1] == Inf] <- 99999999

ada_lasso <-
  cv.glmnet(
    as.matrix(boots_data),
    as.matrix(boots_outcome),
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

index_coef <- sort(union(index_coef,c(1:dim(fof_train_oh)[2])))

##### Searching for the whole variables

selected_var <- row.names(coef)[(index_coef)]

selected_var <- intersect(selected_var,colnames(feature_test_oh))

#### train the model with standard Gaussian prior & LogLoss on test set

select_x_train <- feature_train_oh[, selected_var]

select_x_test <- feature_test_oh[, selected_var]

outcome_temp_test <- outcome_test

for (i in 1:length(outcome_temp_test)) {
  if (outcome_temp_test[i] == -1) {
    outcome_temp_test[i] <- 0
  }
}

LogLoss <- rep(0, day)

beta_prior_mean <- as.matrix(rep(0, length(selected_var)))

beta_prior_var <- diag(1,length(selected_var))

for(i in 1:day){
  range <- c((size*(i-1)+1):(size*i))
  y_star <- as.matrix(select_x_train[range,]) %*% beta_prior_mean + rnorm(size)
  beta_post_var <- base::solve(base::solve(beta_prior_var) + t(select_x_train[range,]) %*% as.matrix(select_x_train[range,]))
  beta_post_mean <- beta_post_var %*% (base::solve(beta_prior_var) %*% beta_prior_mean + t(select_x_train[range,]) %*%  y_star)
  beta_prior_mean <- beta_post_mean
  beta_prior_var <- beta_post_var
  
  Pro <-
    pnorm(outcome_test * ((as.matrix(select_x_test) %*% (beta_post_mean))))
  
  LogLoss[i] <-
    sum(outcome_temp_test * log(Pro) + (1 - outcome_temp_test) * log(1 - Pro)) / (-length(outcome_temp_test))
}

### BLIPBayes
#### Get the EB estimator

LogLoss_BLIP_EB <- rep(0,day)

line <- dim(fof_test_oh)[2] ##### border of first and second features

n_first <- length(intersect(selected_var,colnames(fof_train_oh)))

n_second <- length(selected_var) - n_first

bt <- NULL

for(i in 1 : 12){
  range <- c(1 : size)
  
  bt <- c(bt,sample(x = range, size = length(range),replace = TRUE))
  
}   ##### Bootstrap

##### Compute the meta-prior variance of two categories

beta_prior_mean <- as.matrix(rep(0, length(selected_var)))

beta_prior_var <- diag(1,length(selected_var))

for(i in 1:12){
  range <- bt[(size*(i-1)+1):(size*i)]
  y_star <- as.matrix(select_x_train[range,]) %*% beta_prior_mean + rnorm(size)
  beta_post_var <- base::solve(base::solve(beta_prior_var) + t(select_x_train[range,]) %*% as.matrix(select_x_train[range,]))
  beta_post_mean <- beta_post_var %*% (base::solve(beta_prior_var) %*% beta_prior_mean + t(select_x_train[range,]) %*%  y_star)
  beta_prior_mean <- beta_post_mean
  beta_prior_var <- beta_post_var
}

var_first_EB <- sum((beta_post_mean[c(1:n_first)])^2 - beta_post_var[c(1:n_first)]) / n_first

var_second_EB <- sum((beta_post_mean[c(n_first:length(beta_post_mean))])^2 - beta_post_var[c(n_first:length(beta_post_mean))]) / n_second

##### Restart the BLIP model with new meta-prior

set.seed(1)

beta_prior_mean <- as.matrix(rep(0, length(selected_var)))

beta_prior_var <- as.matrix(diag(c(rep(var_first_EB,n_first),rep(var_second_EB,n_second))))

for(i in 1:day){
  range <- c((size*(i-1)+1):(size*i))
  y_star <- as.matrix(select_x_train[range,]) %*% beta_prior_mean + rnorm(size)
  beta_post_var <- base::solve(base::solve(beta_prior_var) + t(select_x_train[range,]) %*% as.matrix(select_x_train[range,]))
  beta_post_mean <- beta_post_var %*% (base::solve(beta_prior_var) %*% beta_prior_mean + t(select_x_train[range,]) %*%  y_star)
  beta_prior_mean <- beta_post_mean
  beta_prior_var <- beta_post_var
  
  Pro <-
    pnorm(outcome_test * ((as.matrix(select_x_test) %*% (beta_post_mean))))
  
  LogLoss_BLIP_EB[i] <-
    sum(outcome_temp_test * log(Pro) + (1 - outcome_temp_test) * log(1 - Pro)) / (-length(outcome_temp_test))
}


### BLIPTwice
LogLoss_BLIP_EB_twice <- rep(0,day)

line <- dim(fof_test_oh)[2] ##### border of first and second features

n_first <- length(intersect(selected_var,colnames(fof_train_oh)))

n_second <- length(selected_var) - n_first

bt <- NULL

for(i in 1 : 12){
  range <- c(1 : size)
  
  bt <- c(bt,sample(x = range, size = length(range),replace = TRUE))
  
}

bt <- c(bt,c(1:size))

beta_prior_mean <- as.matrix(rep(0, length(selected_var)))

beta_prior_var <- diag(1,length(selected_var))

for(i in 1:13){
  range <- bt[(size*(i-1)+1):(size*i)]
  y_star <- as.matrix(select_x_train[range,]) %*% beta_prior_mean + rnorm(size)
  beta_post_var <- base::solve(base::solve(beta_prior_var) + t(select_x_train[range,]) %*% as.matrix(select_x_train[range,]))
  beta_post_mean <- beta_post_var %*% (base::solve(beta_prior_var) %*% beta_prior_mean + t(select_x_train[range,]) %*%  y_star)
  beta_prior_mean <- beta_post_mean
  beta_prior_var <- beta_post_var
}

var_first_EB <- sum((beta_post_mean[c(1:n_first)])^2 - beta_post_var[c(1:n_first)]) / n_first

var_second_EB <- sum((beta_post_mean[c(n_first:length(beta_post_mean))])^2 - beta_post_var[c(n_first:length(beta_post_mean))]) / n_second

set.seed(1)

beta_prior_mean <- as.matrix(rep(0, length(selected_var)))

beta_prior_var <- as.matrix(diag(c(rep(var_first_EB,n_first),rep(var_second_EB,n_second))))

for(i in 1:day){
  range <- c((size*(i-1)+1):(size*i))
  y_star <- as.matrix(select_x_train[range,]) %*% beta_prior_mean + rnorm(size)
  beta_post_var <- base::solve(base::solve(beta_prior_var) + t(select_x_train[range,]) %*% as.matrix(select_x_train[range,]))
  beta_post_mean <- beta_post_var %*% (base::solve(beta_prior_var) %*% beta_prior_mean + t(select_x_train[range,]) %*%  y_star)
  beta_prior_mean <- beta_post_mean
  beta_prior_var <- beta_post_var
  
  Pro <-
    pnorm(outcome_test * ((as.matrix(select_x_test) %*% (beta_post_mean))))
  
  LogLoss_BLIP_EB_twice[i] <-
    sum(outcome_temp_test * log(Pro) + (1 - outcome_temp_test) * log(1 - Pro)) / (-length(outcome_temp_test))
}

plot(LogLoss,type = "l",ylim = c(0.73,0.82),ylab = "Logloss", xlab = "day", col = "green", main = "Small Batches")
lines(LogLoss_BLIP_EB,type = "l",col = "blue")
lines(LogLoss_BLIP_EB_twice, type = "l",col = "red")
legend(x = 'topleft', 
       legend = c(expression(BLIP),expression(BLIPBayes),expression(BLIPTwice)),
       col = c('green','blue','red'),
       lty = 1)

# The goal of your project is to predict the manner in which they did the exercise. 
# This is the "classe" variable in the training set. 
setwd("/Users/okada/myWork/coursera/practical_machine_learning")
library(lattice);library(ggplot2);library(caret)
trainingall <- read.csv("pml-training.csv"); testing<-read.csv("pml-testing.csv")
dim(trainingall);dim(testing)
# 19622 x 160 data frame, 20 test data
head(trainingall);head(testing)
summary(trainingall);summary(testing)
## some variables are stored as factor, although they are numerical; e.g. kurtosis_picth_belt
## This is caused by "#DIV/0!" values. Replace it with NA from the original data set
## perl -i.bkup -p -e 's/#DIV\/0\!/NA/g;' pml-testing.csv
## perl -i.bkup -p -e 's/#DIV\/0\!/NA/g;' pml-training.csv

## some variables have NA for almost all reacords. For example, kurtosis_picth_belt has 19,248
## NA out of 19,622 records. These variables will not be useful for the classification even if
## we impute the data, because almost all records will have identical values. Variables starting
## with "kurtosis", "skewness", "stdev" seem to have this problem.
## check these variables, and remove them.
## kurtosis_roll_belt, kurtosis_picth_belt, kurtosis_yaw_belt, skewness_roll_belt
## skewness_roll_belt.1, skewness_yaw_belt, max_roll_belt, max_picth_belt, max_yaw_belt
idxMax <- grep("^max_", colnames(trainingall),ignore.case=TRUE)
idxMin <- grep("^min_", colnames(trainingall),ignore.case=TRUE)
idxSkew <- grep("^skewness_", colnames(trainingall),ignore.case=TRUE)
idxKurt <- grep("^kurtosis_", colnames(trainingall),ignore.case=TRUE)
idxStd <- grep("^stddev_", colnames(trainingall),ignore.case=TRUE)
idxVar <- grep("^var_", colnames(trainingall),ignore.case=TRUE)
idxAvg <- grep("^avg_", colnames(trainingall),ignore.case=TRUE)
idxAmp <- grep("^amplitude_", colnames(trainingall),ignore.case=TRUE)

# check if these really have too many NAs
summary(trainingall[,c(idxMax,idxMin,idxSkew,idxKurt)])
summary(trainingall[,c(idxStd,idxVar,idxAvg,idxAmp)])
# check what's left have numerical values
summary(trainingall[,-c(idxMax,idxMin,idxSkew,idxKurt,idxStd,idxVar,idxAvg,idxAmp)])
## OK
trainingall <- trainingall[,-c(1,2,3,4,5,idxMax,idxMin,idxSkew,idxKurt,idxStd,idxVar,idxAvg,idxAmp)]
testing <- testing[,-c(1,2,3,4,5,idxMax,idxMin,idxSkew,idxKurt,idxStd,idxVar,idxAvg,idxAmp)]
dim(trainingall);dim(testing)
## now we have 60 features, drop from 160 variables.

qplot(roll_belt, pitch_belt, data=trainingall, colour=classe)
qplot(roll_belt, yaw_belt, data=trainingall, colour=classe)

## variables with high correlations for numerical variables
tmp <- trainingall[,-c(1,2,3,4,5,6,60)]
descrCor <- cor(tmp)
highlyCor <- findCorrelation(descrCor,cutoff=0.90)
colnames(tmp)[highlyCor]
cor(tmp[, grep("^accel_belt", colnames(tmp))])
cor(tmp[, grep("^gyros_arm", colnames(tmp))])
cor(tmp[, grep("^gyros_dumbbell", colnames(tmp))])

findLinearCombos(tmp)

## select a few variables to build a model
## split the training data set into training and cv data sets.
#set.seed(11)
set.seed(39104)
inTrain <- createDataPartition(y=trainingall$classe, p=0.7, list=FALSE)
training <- trainingall[inTrain,];cv <- trainingall[-inTrain,]
dim(training);dim(cv)

## cross validation 
control <- trainControl (method = "cv", number = 3, repeats = 1)


#modFit <- train(classe~magnet_forearm_x+roll_belt+yaw_belt, data=train, method="rpart", prox=TRUE)
modFit <- train(classe~., data=training, method="rpart", trControl = control)
predictions <- predict(modFit, cv)
confusionMatrix(predictions, cv$classe)

modFitRF <- train(classe~., data=training, method="rf", trControl = control)
predrf <- predict(modFitRF, cv)
confusionMatrix(predrf, cv$classe)
save.image("pml.RData")

plot(varImp (modFitRF, scale = FALSE), top = 20)
trellis.par.set(caretTheme())
plot (modFitRF, type = c("g", "o"))
#modFitGBM <- train(classe~., data=training, method="gbm",n.tree=100, shrinkage=0.01,
#                   interaction.depth=1, trControl = control)
modFitGBM <- train(classe~., data=training, method="gbm",trControl = control)

print(modFitGBM)
summary(modFitGBM$finalModel)
predGBM <- predict(modFitGBM, cv)
confusionMatrix(predGBM, cv$classe)

# which model parameters were most effective and ultimately selected for the final mode.
trellis.par.set (caretTheme())
plot (modFitGBM, type = c("g", "o"))

plot(varImp (modFitGBM, scale = FALSE), top = 20)

########################################################
## Pre Processing
########################################################
preObj <- preProcess(training[,-160],method="knnImpute")
capAve <- predict(preObj,training[,-160])$capAve



########################################################
## Fit the model to the testing data set
########################################################

## benchmark
table(training$classe)
## Class A has the highest frequecy. The benchmark sets all the prediction as A
pred.benchmark <- factor(rep("A",dim(cv)[1]), levels=c("A","B","C","D","E"))
confusionMatrix(pred.benchmark, cv$classe)
## accuracy = 28.45%

###########################################
### Fit the model to the test data set
### 
predrf.test <- predict(modFitRF, testing)

# answers = rep("A", 20)
answers <- predrf.test
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(as.character(answers))


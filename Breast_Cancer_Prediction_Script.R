#Loading important packages
library(tidyverse)
library(nclSLR)
library(mlbench)  
library(leaps)
library(varhandle)
library(bestglm)
library(MASS)

#Data acquisition
data(BreastCancer) 
## Check size 
dim(BreastCancer)
# Print first few rows
head(BreastCancer)
names(BreastCancer)

#------------------------------------------------------------------------------------------------
#Data Wrangling
#convert to factors to quantitative variables
class(BreastCancer$Class)
typeof(BreastCancer$Class)
head(BreastCancer$Class) #benign is the first level and malignant the second


#convert factors to numeric variables without changing values, changing Class to integer vector with two levels, removing ID column
MyData=data.frame(unfactor(BreastCancer[,2:10]),Class=as.integer(BreastCancer$Class)-1) 
MyData=na.omit(MyData) #removing rows with NA values

# Extract response variable:
y=MyData[,10]
# Extract the predictor variables:
X1_raw= MyData[,1:9]
#Standardise the predictor variables: 
X1=scale(as.matrix(X1_raw))
#Reformed data frame
(BC_data=data.frame(X1,y))

# Store n and p
n = nrow(BC_data)
p = ncol(BC_data) - 1

#------------------------------------------------------------------------------------------------
#Graphical Summary
#Scatter plot
pairs(MyData[,1:9],col=MyData[,10]+1)
#Almost all variables the ratings given to benign is low - healthier 
#Linear correlation between cell size and cell shape, the bigger the cell size the more irregular the cell shape

#Numerical Summary
library(dplyr)
#Variable means of benign samples 
MyData %>% filter(MyData$Class==0) %>% colMeans()

#Variable means of malignant samples
MyData %>% filter(MyData$Class==1) %>% colMeans()
#The means for all the predictor variables with malignant samples are higher than the means for benign samples
#This suggests that malignant cells are very unhealthy looking and rogue 

#Measure of scatter
#Since all the variables have been measured on a scale of 1-10 no need for standardization of data 
v=var(MyData[,1:9])
diag(v)
#Bare.nuclei is highly spread compared to the other variables. Mitoses has the smallest variance suggesting that it is more centered around the mean compared to the other variables

#Correlation Matrix - quantify the strength 
cor(MyData[,1:9])
#Cell.size and Cell.shape are highly correlated (0.907)

#Covariance matrix of standardized data
var(X1)
#Standardized data covariance matrix is the same as correlation matrix of original data 

#------------------------------------------------------------------------------------------------
#Best subset selection with logistic regression

#Apply best subset selection
bss_fit_AIC=bestglm(BC_data, family=binomial, IC="AIC")
bss_fit_BIC=bestglm(BC_data, family=binomial, IC="BIC")

#Examining results
bss_fit_AIC$Subsets
(best_AIC=bss_fit_AIC$ModelReport$Bestk)

bss_fit_BIC$Subsets
(best_BIC=bss_fit_BIC$ModelReport$Bestk)

# Create multi-panel plotting device
par(mfrow=c(1,2))
## Produce plots, highlighting optimal value of k
plot(0:p, bss_fit_AIC$Subsets$AIC, xlab="Number of predictors", ylab="AIC", type="b") 
points(best_AIC, bss_fit_AIC$Subsets$AIC[best_AIC+1], col="red", pch=16)
plot(0:p, bss_fit_BIC$Subsets$BIC, xlab="Number of predictors", ylab="BIC", type="b")
points(best_BIC, bss_fit_BIC$Subsets$BIC[best_BIC+1], col="red", pch=16)
#Resetting panel
par(mfrow=c(1,1))

#Model with 6 predictor variables looks like a good compromise

#Selecting best model
pstar=6
#Check which predictors are in the 6 predictor model
bss_fit_AIC$Subsets[pstar+1,]
#Construct a reduced data set containing only the selected predictor
(indices=as.logical(bss_fit_AIC$Subsets[pstar+1,2:(p+1)]))
BC_data_red = data.frame(X1[,indices], y)

#Obtain regression coefficients for this model
glm_fit=glm(y~.,data=BC_data_red,family="binomial")
summary(glm_fit)
#All the regression coefficients show significance

  



#---------------------------------------------------------------------------------------------------------------------
#Regularized logistic regression with LASSO penalty

library(glmnet)# Load the glmnet package
# Choosing grid of values for the tuning parameter
grid = 10^seq(-4,1, length.out=100)
# Fitting a model with LASSO penalty for each value of the tuning parameter
lasso_fit = glmnet(X1, y, family="binomial", alpha=1, standardize=FALSE, lambda=grid)

# Examine the effect of the tuning parameter on the parameter estimates
plot(lasso_fit, xvar="lambda", col=rainbow(p), label=TRUE)

#To select a single value for the tuning parameter and perform K-fold cross-validation
lasso_cv_fit = cv.glmnet(X1, y, family="binomial", alpha=1, standardize=FALSE, lambda=grid, type.measure="class",foldid = fold_index )
plot(lasso_cv_fit) #Cross validation scores

# Identifying the optimal value for the tuning parameter
(lambda_lasso_min = lasso_cv_fit$lambda.min)
which_lambda_lasso = which(lasso_cv_fit$lambda == lambda_lasso_min)

#Test error for LASSO
lasso_cv_fit$cvm[which_lambda_lasso] 

# The parameter estimates associated with optimal value of the tuning parameter 
coef(lasso_fit, s=lambda_lasso_min)
#None of the variables were eliminated as none shrunk to zero completely
#Perform ridge regression and test which model has lower test error 

#---------------------------------------------------------------------------------------------------------------------
#Regularized logistic regression with Ridge penalty
# Choosing grid of values for the tuning parameter
grid2 = 10^seq(5,-3, length.out=100)
# Fitting a model with LASSO penalty for each value of the tuning parameter
ridge_fit = glmnet(X1, y, family="binomial", alpha=0, standardize=FALSE, lambda=grid2)

#Examine the effect of the tuning parameter on the parameter estimates
plot(ridge_fit, xvar="lambda", col=rainbow(p), label=TRUE)


#To select a single value for the tuning parameter
ridge_cv_fit = cv.glmnet(X1, y, family="binomial", alpha=0, standardize=FALSE, lambda=grid,type.measure="class",foldid = fold_index)
plot(lasso_cv_fit)

# Identify the optimal value for the tuning parameter
(lambda_ridge_min = ridge_cv_fit$lambda.min)
which_lambda_ridge = which(ridge_cv_fit$lambda == lambda_ridge_min)

#Test error Ridge Regression
ridge_cv_fit$cvm[which_lambda_ridge]

# Find the parameter estimates associated with optimal value of the tuning parameter 
coef(ridge_fit, s=lambda_ridge_min)


#----------------------------------------------------------------------------------------------------------------------
#LDA
#Perform LDA on dat with 6 variables significant variables from best subset selection 
(lda_fit=lda(y~.,data=BC_data_red))


#Function to estimate average misclassification error by general K-fold cross validation
lda_cv = function(X1, y, fold_ind) {
  Xy = data.frame(X1, y=y)
  nfolds = max(fold_ind)
  if(!all.equal(sort(unique(fold_ind)), 1:nfolds)) stop("Invalid fold partition.") 
  cv_errors = numeric(nfolds)
  for(fold in 1:nfolds) {
    tmp_fit = lda(y ~ ., data=Xy[fold_ind!=fold,]) 
    lda_test = predict(tmp_fit, Xy[fold_ind==fold,]) 
    yhat= lda_test$class
    yobs = y[fold_ind==fold]
    cv_errors[fold] = 1- mean(yobs==yhat)
  }
  fold_sizes = numeric(nfolds)
  for(fold in 1:nfolds) fold_sizes[fold] = length(which(fold_ind==fold))
  test_error = weighted.mean(cv_errors, w=fold_sizes)
  return(test_error)
}
#Test error for Model with 6 variables with LDA
lda_cv(X1[,indices], y,fold_index)
#----------------------------------------------------------------------------------------------------------------------
#QDA
#Perform QDA on data with 6 variables significant variables from best subset selection 
(qda_fit=qda(y~.,data=BC_data_red))
#Function to estimate average misclassification error by general K-fold cross validation
qda_cv = function(X1, y, fold_ind) {
  Xy = data.frame(X1, y=y)
  nfolds = max(fold_ind)
  if(!all.equal(sort(unique(fold_ind)), 1:nfolds)) stop("Invalid fold partition.") 
  cv_errors = numeric(nfolds)
  for(fold in 1:nfolds) {
    tmp_fit = qda(y ~ ., data=Xy[fold_ind!=fold,]) 
    qda_test = predict(tmp_fit, Xy[fold_ind==fold,]) 
    yhat= qda_test$class
    yobs = y[fold_ind==fold]
    cv_errors[fold] = 1- mean(yobs==yhat)
  }
  fold_sizes = numeric(nfolds)
  for(fold in 1:nfolds) fold_sizes[fold] = length(which(fold_ind==fold))
  test_error = weighted.mean(cv_errors, w=fold_sizes)
  return(test_error)
}
#Test error for Model with 6 variables with LDA
qda_cv(X1[,indices], y,fold_index)



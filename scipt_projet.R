#libraries

install.packages("leaps") # Best Subset Selection
library(leaps)


# code Subset Selection Method

housing<-read.csv("R/Projet R/traitement_housing.csv",sep=";")
head(housing); View(housing) ; print(names(housing)) ; print(dim(housing))
sum(is.na(housing)) # 0 !
reglin<-lm(MEDV~.,housing)
summary(reglin) # r squared of 74% + low pval(Fstat) 
plot(reglin)

predictors<-log(housing$DIS+1)+log(housing$INDUS+1)+log(housing$CRIM+1)+log(housing$ZN+1)+log(housing$Chase+1)+log(housing$NOX+1)+log(housing$RM+1)+log(housing$AGE+1)+log(housing$RAD+1)+log(housing$TAX+1)+log(housing$PTRATIO+1)+log(housing$B+1)+log(housing$LTSAT+1)
logreg <- lm(log(MEDV+1)~predictors,housing)
logreg2<-loglm(MEDV~predictors,housing)
plot(logreg)
summary(logreg)



regfit.full<-regsubsets(MEDV~.,housing,nvmax=13)
regsummary=summary(regfit.full) 


# show for each subset size, the variables that should be kept to have the best model (smallest RSS)
regsummary
names(regsummary) # different methods of selection of the best model between the differents subset sizes

par(mfrow=c(3,2))
plot(regsummary$rss,xlab="nb of variables",ylab="RSS",type="l")
plot(regsummary$rsq,xlab="nb of variables",ylab="R squared",type="l")
plot(regsummary$adjr2,xlab="nb of variables",ylab="adjusted R squared",type="l")
# no big difference between r2 and adjusted r2

plot(regsummary$cp,xlab="nb of variables",ylab="Cp", type="l")
plot(regsummary$bic,xlab="number of variables",ylab="BIC",type="l")

which.min(regsummary$rss)
which.max(regsummary$rsq)
#same result of 13 (the number of explanatory variables) it is consistent with the fact that it always increases/decreases with the number of variables

which.max(regsummary$adjr2) # 11

which.min(regsummary$cp)
which.min(regsummary$bic)
#we obtain the same result (and same as adjr2) : the model with 11 explanatory variables

points(11,regsummary$bic[11],col="red",cex=2,pch=20)

newregfit<-lm(MEDV~CRIM+ZN+Chase+NOX+RM+DIS+RAD+TAX+PTRATIO+B+LTSAT,housing) # new regression with the 11 remaining variables
summary(newregfit)
summary(lm(MEDV~.,housing)) # just to compare both
# while reducing the number of variables, the level of significance of the remaining variables is quite the same as the model with all the predictors, except for the predictor "TAX" which has become more significant
# the multiple r^2 remains unchanged
# the adjusted r^2 slightly increases from 73,38% to 73,48% (linked to the "penalty")

coef(regfit.full,11)

# Forward/Bacward Stepwsise Regression

regfit.fwd<-regsubsets(MEDV~.,housing,nvmax=13,method="forward")
summary(regfit.fwd)
which.min(summary(regfit.fwd)$bic)
regfit.bwd<-regsubsets(MEDV~.,housing,nvmax=13,method="backward")
summary(regfit.bwd)
which.min(summary(regfit.bwd)$bic)
# we obtain the same result as with the Subset Selection Method

# Cross-Validation

predict.regsubsets=function(object,newdata,id,...){
  form=as.formula(object$call[[2]])
  mat=model.matrix(form,newdata)
  coefi=coef(object,id=id)
  xvars=names(coefi)
  mat[,xvars]%*%coefi
}
regfit.best=regsubsets(MEDV~.,data=housing,nvmax=13)
# ## c.v.


k=15 # ou autre ...
for(k in 1:30)
set.seed(8)
# create a vector that allocates each obs to one of the k=10 folds
folds=sample(1:k,nrow(housing),replace=TRUE)
cv.errors=matrix(NA,k,13, dimnames=list(NULL, paste(1:13)))
for(j in 1:k){
  best.fit=regsubsets(MEDV~.,data=housing[folds!=j,],nvmax=13) # estimate outside the fold j
  for(i in 1:13){
    pred=predict(best.fit,housing[folds==j,],id=i) 
    cv.errors[j,i]=mean((housing$MEDV[folds==j]-pred)^2) 
  }
}
mean.cv.errors=apply(cv.errors,2,mean) 
# mean across 13 models (k=10 folds, could be done more than once!)
mean.cv.errors
which.min(mean.cv.errors)


par(mfrow=c(1,1))
# using k=10 > the model with 11 predictors should be kept according to this method, however the means (of the SSR values obtain in the 10 subsets) are close, therefore we could reasonably think that using a penalty, we would obtain a lower number of predictors
plot(mean.cv.errors,type='b')
reg.best=regsubsets(MEDV~.,data=housing, nvmax=13)
coef(reg.best,11) 

# Ridge Regression Code

install.packages("glmnet")
library(glmnet)
# package for elastic net

x=model.matrix(MEDV~.,housing)[,-1]
y=housing$MEDV
grid=10^seq(10,-2,length=121)
ridge.mod=glmnet(x,y,alpha=0,lambda=grid)
dim(coef(ridge.mod))

ridge.mod$lambda[50]
coef(ridge.mod)[,50]
sqrt(sum(coef(ridge.mod)[-1,50]^2))

ridge.mod$lambda[60]
coef(ridge.mod)[,60]
sqrt(sum(coef(ridge.mod)[-1,60]^2))

predict(ridge.mod,s=50,type="coefficients")[1:14,]
predict(ridge.mod,s=60,type="coefficients")[1:14,]

# Train and validate

set.seed(121)
train=sample(1:nrow(x),nrow(x)/2) # 2 datasets of same size
test=(-train)
y.test=y[test]
# first try with lambda=4
ridge.mod=glmnet(x[train,],y[train],alpha=0,lambda=grid,thresh=1e-12) # we calibrate on the "train" set of data
ridge.pred=predict(ridge.mod,s=4,newx=x[test,]) # we predict on the "test" set using the "train" calibration to predict
mean((ridge.pred-y.test)^2) # out of sample error
mean((mean(y[train])-y.test)^2) # error relative to mean
# then we can try with very high lambda, as lambda increases, the coefficient are drived toward zero, therefore the dependant variable should be close to the mean
ridge.pred=predict(ridge.mod,s=1e10,newx=x[test,])
mean((ridge.pred-y.test)^2) 
# as forecasted, it is equal to the precedent result. Indeed, when lamba is very high (~infinite penalty), the betas of the regresssion calibrated on "train" are close to 0, so the predicted variables are merely equals to the intercept, meaning the mean of the "train" dataset, in other words ridge.pred=mean(y[train])
ridge.pred=predict(ridge.mod,s=0,newx=x,x=x,y=y,exact=T)
mean((ridge.pred-y.test)^2) 
lm(y~x,subset=train)
predict(ridge.mod,s=0,exact=T,x=x,y=y,type="coefficients")[1:14,]
# pas compris

# With the Cross Validation Method

set.seed(121)
cv.out=cv.glmnet(x[train,],y[train],alpha=0)
plot(cv.out)
bestlambda=cv.out$lambda.min
bestlambda
ridge.pred=predict(ridge.mod,s=bestlambda,newx=x[test,])
mean((ridge.pred-y.test)^2)
out=glmnet(x,y,alpha=0)
predict(out,type="coefficients",s=bestlambda)[1:14,]




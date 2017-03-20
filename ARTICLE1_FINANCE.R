setwd("/Users/Alex")
require(lmtest)
dataSet = read.csv('forR.csv')

X_ <- na.omit(dataSet)
X.Y = as.matrix(X_[1])
X.sub = as.matrix(cbind(X_[6],X_[4],X_[5], X_[8], X_[7], X_[2]))
X.full = as.matrix(cbind(X_[6],X_[4],X_[5], X_[8], X_[7], X_[2], X_[3]))

fm1 <- lm(X.Y ~ X.sub)
fm2 <- lm(X.Y ~ X.full)
lrtest(fm2, fm1)

#2, 6, 5
#NOT7, 8 4, 3
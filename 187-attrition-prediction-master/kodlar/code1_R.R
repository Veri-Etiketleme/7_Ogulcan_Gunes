setwd("/users/wangzhe/Desktop/course material/business analytics with R/data")
eadata<-read.csv("EmployeeAttrition.csv")

# exam missingvalue 
sum(complete.cases(eadata))
sum(!complete.cases(eadata))
mean(!complete.cases(eadata))
eadata[!complete.cases(eadata),]

# exam the outliers
outlier_values <- boxplot.stats(eadata$Age)$out 
boxplot(eadata$Age, main="Age", boxwex=1)
mtext(paste("Outliers: ", paste(outlier_values, collapse=", ")), cex=0.8)

outlier_values <- boxplot.stats(eadata$DailyRate)$out 
boxplot(eadata$DailyRate, main="DailyRate", boxwex=1)
mtext(paste("Outliers: ", paste(outlier_values, collapse=", ")), cex=0.8)

outlier_values <- boxplot.stats(eadata$DistanceFromHome)$out 
boxplot(eadata$DistanceFromHome, main="DistanceFromHome", boxwex=1)
mtext(paste("Outliers: ", paste(outlier_values, collapse=", ")), cex=0.8)

outlier_values <- boxplot.stats(eadata$HourlyRate)$out 
boxplot(eadata$HourlyRate, main="HourlyRate", boxwex=1)
mtext(paste("Outliers: ", paste(outlier_values, collapse=", ")), cex=0.8)

outlier_values <- boxplot.stats(eadata$JobInvolvement)$out 
boxplot(eadata$JobInvolvement, main="JobInvolvement", boxwex=1)
mtext(paste("Outliers: ", paste(outlier_values, collapse=", ")), cex=0.8)

outlier_values <- boxplot.stats(eadata$JobLevel)$out 
boxplot(eadata$JobLevel, main="JobLevel", boxwex=1)
mtext(paste("Outliers: ", paste(outlier_values, collapse=", ")), cex=0.8)

outlier_values <- boxplot.stats(eadata$MonthlyIncome)$out 
boxplot(eadata$MonthlyIncome, main="MonthlyIncome", boxwex=1)
mtext(paste("Outliers: ", paste(outlier_values, collapse=", ")), cex=0.8)

outlier_values <- boxplot.stats(eadata$MonthlyRate)$out 
boxplot(eadata$MonthlyRate, main="MonthlyRate", boxwex=1)
mtext(paste("Outliers: ", paste(outlier_values, collapse=", ")), cex=0.8)

outlier_values <- boxplot.stats(eadata$NumCompaniesWorked)$out 
boxplot(eadata$NumCompaniesWorked, main="NumCompaniesWorked", boxwex=1)
mtext(paste("Outliers: ", paste(outlier_values, collapse=", ")), cex=0.8)

outlier_values <- boxplot.stats(eadata$PercentSalaryHike)$out 
boxplot(eadata$PercentSalaryHike, main="PercentSalaryHike", boxwex=1)
mtext(paste("Outliers: ", paste(outlier_values, collapse=", ")), cex=0.8)

outlier_values <- boxplot.stats(eadata$PerformanceRating)$out 
boxplot(eadata$PerformanceRating, main="PerformanceRating", boxwex=1)
mtext(paste("Outliers: ", paste(outlier_values, collapse=", ")), cex=0.8)

outlier_values <- boxplot.stats(eadata$RelationshipSatisfaction)$out 
boxplot(eadata$RelationshipSatisfaction, main="RelationshipSatisfaction", boxwex=1)
mtext(paste("Outliers: ", paste(outlier_values, collapse=", ")), cex=0.8)

outlier_values <- boxplot.stats(eadata$TotalWorkingYears)$out 
boxplot(eadata$TotalWorkingYears, main="TotalWorkingYears", boxwex=1)
mtext(paste("Outliers: ", paste(outlier_values, collapse=", ")), cex=0.8)

outlier_values <- boxplot.stats(eadata$TrainingTimesLastYear)$out 
boxplot(eadata$TrainingTimesLastYear, main="TrainingTimesLastYear", boxwex=1)
mtext(paste("Outliers: ", paste(outlier_values, collapse=", ")), cex=0.8)

outlier_values <- boxplot.stats(eadata$WorkLifeBalance)$out 
boxplot(eadata$WorkLifeBalance, main="WorkLifeBalance", boxwex=1)
mtext(paste("Outliers: ", paste(outlier_values, collapse=", ")), cex=0.8)

outlier_values <- boxplot.stats(eadata$YearsAtCompany)$out 
boxplot(eadata$YearsAtCompany, main="YearsAtCompany", boxwex=1)
mtext(paste("Outliers: ", paste(outlier_values, collapse=", ")), cex=0.8)

outlier_values <- boxplot.stats(eadata$YearsInCurrentRole)$out 
boxplot(eadata$YearsInCurrentRole, main="YearsInCurrentRole", boxwex=1)
mtext(paste("Outliers: ", paste(outlier_values, collapse=", ")), cex=0.8)

outlier_values <- boxplot.stats(eadata$YearsSinceLastPromotion)$out 
boxplot(eadata$YearsSinceLastPromotion, main="YearsSinceLastPromotion", boxwex=1)
mtext(paste("Outliers: ", paste(outlier_values, collapse=", ")), cex=0.8)

outlier_values <- boxplot.stats(eadata$YearsWithCurrManager)$out 
boxplot(eadata$YearsWithCurrManager, main="YearsWithCurrManager", boxwex=1)
mtext(paste("Outliers: ", paste(outlier_values, collapse=", ")), cex=0.8)

#Choose data
eadata=eadata[eadata$JobLevel<=4,]

# logistic regression
set.seed(42)

eadata$EmployeeCount<-NULL
eadata$DailyRate<-NULL
eadata$HourlyRate<-NULL
eadata$Over18<-NULL
eadata$StandardHours<-NULL
eadata$EmployeeNumber<-NULL


trainfrac<-0.7
testfrac<-0.3
sampleSizeTraining<-floor(trainfrac*nrow(eadata))
sampleSizeTest <- floor(testfrac * nrow(eadata))
indicesTraining <- sort(sample(seq_len(nrow(eadata)), size=sampleSizeTraining))
indicesTest <- setdiff(seq_len(nrow(eadata)), indicesTraining)
eadatatrain<-eadata[indicesTraining,]
eadatatest<-eadata[indicesTest,]

ealogit<-glm(Attrition~.,data = eadatatrain,family = binomial(link = "logit"))
summary(ealogit)


#Test Model Performance 
testModelPerformance <- function(model, dataset, target, prediction) {
  if(missing(prediction))
  {
    print("here")
    dataset$pred <- predict(model, dataset, type = "class")
  }
  else
  {
    print("here2")
    dataset$pred <- prediction
  }
  
  writeLines("PERFORMANCE EVALUATION FOR")
  writeLines(paste("Model:", deparse(substitute(model))))
  writeLines(paste("Target:", deparse(substitute(target))))
  
  writeLines("\n\nConfusion Matrix:")
  confMatrix <- table(Actual = target, Predicted = dataset$pred)
  truePos <- confMatrix[2,2]
  falseNeg <- confMatrix[2,1]
  falsePos <- confMatrix[1,2]
  trueNeg <- confMatrix[1,1]
  print(confMatrix)
  writeLines("\n\n")
  
  accuracy <- (truePos + trueNeg)/(truePos + falseNeg + falsePos + trueNeg)
  sensitivity <- truePos/(truePos + falseNeg)
  specificity <- trueNeg/(falsePos + trueNeg)
  falsePosRate <- falsePos/(falsePos + trueNeg)
  falseNegRate <- falseNeg/(truePos + falseNeg)
  precision <- truePos/(truePos + falsePos)
  
  writeLines(paste("Accuracy:", round(accuracy, digits = 4)))
  writeLines(paste("Sensitivity:", round(sensitivity, digits = 4)))
  writeLines(paste("Specificity:", round(specificity, digits = 4)))
  writeLines(paste("False Positive Rate:", round(falsePosRate, digits = 4)))
  writeLines(paste("False Negative Rate:", round(falseNegRate, digits = 4)))
  writeLines(paste("Precision:", round(precision, digits = 4)))
  
  dataset
}

ealogit<-step(ealogit)
summary(ealogit)
anova(object = ealogit, test = 'Chisq')

eadatatrain$prob <- predict(ealogit,newdata = eadatatrain,type = "response")
eadatatrain$logitpred <- round(eadatatrain$prob)
eadatatrain <- testModelPerformance(ealogit, eadatatrain, eadatatrain$Attrition, eadatatrain$logitpred)
eadatatest$prob <- predict(ealogit, newdata = eadatatest, type = "response")
eadatatest$logitpred <- round(eadatatest$prob)
eadatatest <-testModelPerformance(ealogit,eadatatest,eadatatest$Attrition,eadatatest$logitpred)

library(gplots)
library(ROCR)
library(ROCR)
p <- predict(ealogit, newdata=eadatatest, type="response")
pr <- prediction(p, eadatatest$Attrition)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf)

library(pROC)
roc_curve <- roc(eadatatest$Attrition,p)
names(roc_curve)
x <- 1-roc_curve$specificities
y <- roc_curve$sensitivities
library(ggplot2)
p <- ggplot(data = NULL, mapping = aes(x= x, y = y))
p + geom_line(colour = 'red') +geom_abline(intercept = 0, slope = 1)+annotate('text', x = 0.4, y = 0.5, label =paste('AUC=',round(roc_curve$auc,2)))+labs(x = '1-specificities',y = 'sensitivities', title = 'ROC Curve')

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc


library(BaylorEdPsych)
PseudoR2(ealogit)
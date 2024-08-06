library(caret)
library(e1071)
library(MASS)
library(randomForest)
library(rpart)
library(ROCR)
library(pROC)
library(ggplot2)
library(data.table)  # For fread, a faster read.csv alternative
library(MLmetrics)

# Load the data
df1 <- fread('data/cleaned.csv')
# Select features
features <- c('host_response_time', 'host_response_rate', 'host_is_superhost',
              'room_type', 'accommodates', 'minimum_nights', 'availability_30',
              'availability_60', 'availability_90', 'availability_365',
              'review_scores_accuracy', 'review_scores_cleanliness',
              'review_scores_checkin', 'review_scores_communication',
              'review_scores_value', 'instant_bookable', 'reviews_per_month',
              'host_years', 'review_years_range')
X <- df1[, ..features, with = FALSE]
df1$popularity <- factor(df1$popularity)
levels(df1$popularity) <- make.names(levels(df1$popularity))
y <- df1$popularity
# Normalize features
preProcValues <- preProcess(X, method = c("center", "scale"))
X_normalized <- predict(preProcValues, X)

# Split the data
set.seed(42)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X_normalized[trainIndex, ]
X_test <- X_normalized[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# Set up control for train() function
fitControl <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = multiClassSummary)

# Logistic Regression
grid_lr <- expand.grid(.alpha = c(0, 0.5, 1),   
                       .lambda = c(0.001, 0.01, 0.1))

# SVM (Using linear and radial basis kernels for illustration)
grid_svm <- expand.grid(.sigma = c(0.01, 0.1),  # Only used in radial basis function
                        .C = c(0.1, 0.5, 1))

# LDA
grid_lda <- expand.grid(.method = c("moment", "mle"))  # Placeholder, caret does not use this syntax for LDA

# QDA
grid_qda <- expand.grid(.method = c("moment", "mle"))  # Placeholder, caret does not use this syntax for QDA

# Random Forest
grid_rfc <- expand.grid(.mtry = c(2, 5, 7),
                        .splitrule = "gini",
                        .min.node.size = 1)

# Decision Tree
grid_tree <- expand.grid(.cp = c(0.01, 0.05, 0.1))

train_and_evaluate <- function(model_type, grid, train_data, train_labels, test_data, test_labels) {
  model <- train(x = train_data, y = train_labels, method = model_type,
                 trControl = fitControl, tuneGrid = grid)
  
  # Predictions and probabilities
  predictions <- predict(model, newdata = test_data)
  probabilities <- predict(model, newdata = test_data, type = "prob")
  
  # Metrics
  confusion <- confusionMatrix(predictions, test_labels)
  roc_curve <- roc(test_labels, probabilities[,2])  # Adjust depending on levels
  auc_score <- auc(roc_curve)
  
  # Plotting
  plot_confusion_and_roc(confusion$table, roc_curve)
  
  list(model = model, auc = auc_score, accuracy = confusion$overall['Accuracy'],
       precision = confusion$byClass['Precision'], recall = confusion$byClass['Recall'],
       f1 = confusion$byClass['F1'], support = confusion$byClass['Pos Pred Value'])
}

# Plot function for Confusion Matrix and ROC Curve
plot_confusion_and_roc <- function(confusion, roc_curve) {
  op <- par(mfrow = c(1, 2), mar = c(4, 4, 2, 1))
  # Confusion Matrix
  heatmap(as.matrix(confusion), Rowv = NA, Colv = NA, scale = "none", 
          labRow = levels(y_test), labCol = levels(y_test), xlab = "Predicted", ylab = "True")
  
  # ROC Curve
  plot(roc_curve, main = "ROC Curve", col = "#1c61b6")
  abline(0, 1, lty = 2, col = "red")
  par(op)
}

# Logistic Regression
results_lr <- train_and_evaluate("glmnet", grid_lr, X_train, y_train, X_test, y_test)
print(paste("AUC Score:", results_lr$auc))
print(paste("Accuracy:", results_lr$accuracy))
print(paste("Precision:", results_lr$precision))
print(paste("Recall:", results_lr$recall))
print(paste("F1 Score:", results_lr$f1))
print(paste("Support:", results_lr$support))

# SVM
results_svm <- train_and_evaluate("svmRadial", grid_svm, X_train, y_train, X_test, y_test)
print("SVM Results:")
print(paste("AUC Score:", results_svm$auc))
print(paste("Accuracy:", results_svm$accuracy))
print(paste("Precision:", results_svm$precision))
print(paste("Recall:", results_svm$recall))
print(paste("F1 Score:", results_svm$f1))
print(paste("Support:", results_svm$support))

# LDA
results_lda <- train_and_evaluate("lda", data.frame(), X_train, y_train, X_test, y_test)
print("LDA Results:")
print(paste("AUC Score:", results_lda$auc))
print(paste("Accuracy:", results_lda$accuracy))
print(paste("Precision:", results_lda$precision))
print(paste("Recall:", results_lda$recall))
print(paste("F1 Score:", results_lda$f1))
print(paste("Support:", results_lda$support))




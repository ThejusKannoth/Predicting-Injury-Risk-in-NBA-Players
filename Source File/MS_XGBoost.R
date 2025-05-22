library(xgboost)
library(caret)
library(Matrix)
setwd('C:/Users/matts/Downloads')
df = read.csv('nba_injured_players.csv', sep=',')

###############################################################################

data <- df[, -c(1,2,3,4,20,28,29,30)]

data$INJURED_TYPE <- as.factor(data$INJURED_TYPE)

###############################################################################

trainIndex <- createDataPartition(data$INJURED_TYPE, p = 0.8, list = FALSE)

train_dataData <- data[trainIndex, ]

test_dataData <- data[-trainIndex, ]


train_labels <- as.numeric(train_dataData$INJURED_TYPE) - 1  # Convert factor levels to 0-based index
test_labels  <- as.numeric(test_dataData$INJURED_TYPE) - 1  

# Convert data to a sparse matrix
train_matrix <- sparse.model.matrix(INJURED_TYPE ~ . -1, data = train_dataData)
test_matrix  <- sparse.model.matrix(INJURED_TYPE ~ . -1, data = test_dataData)


train_features <- colnames(train_matrix)
test_features  <- colnames(test_matrix)

# Check for any differences
setdiff(train_features, test_features)  # Features in training but not in testing
setdiff(test_features, train_features)

missing_cols <- setdiff(train_features, test_features)

# Add missing columns to test matrix with 0 values
for (col in missing_cols) {
  test_matrix <- cbind(test_matrix, 0)
  colnames(test_matrix)[ncol(test_matrix)] <- col
}

# Reorder columns to match training set
test_matrix <- test_matrix[, train_features]

identical(colnames(train_matrix), colnames(test_matrix))


# Check for NA values in labels
sum(is.na(train_labels))  # Should be 0
sum(is.na(test_labels))   # Should be 0

# Check for infinite values
sum(is.infinite(train_labels))  # Should be 0
sum(is.infinite(test_labels))   # Should be 0

# Check for extreme values
range(train_labels)  # Should be within valid range
range(test_labels)



dtrain <- xgb.DMatrix(data = train_matrix, label = train_labels)
dtest  <- xgb.DMatrix(data = test_matrix, label = test_labels)

###############################################################################


xgb_params <- list(
  objective = "multi:softmax",  # Multi-class classification
  num_class = length(unique(train_labels)),  # Number of injury types
  eval_metric = "mlogloss",  # Multi-class log loss
  eta = 0.1,  # Learning rate
  max_depth = 6,  # Depth of trees
  subsample = 0.8,  # Subsample ratio to reduce overfitting
  colsample_bytree = 0.8  # Feature sampling
)

# Train the model
set.seed(123)
xgb_model <- xgb.train(params = xgb_params, 
                       data = dtrain, 
                       nrounds = 100,  # Number of boosting rounds
                       watchlist = list(train = dtrain, test = dtest),
                       early_stopping_rounds = 10,  # Stop if no improvement
                       print_every_n = 10)

# Print model summary
print(xgb_model)


xgb_predictions <- predict(xgb_model, newdata = dtest)


# Convert predictions back to factor labels

xgb_predictions <- factor(xgb_predictions, levels = 0:(length(unique(train_labels)) - 1), 
                          labels = levels(train_dataData$INJURED_TYPE))



conf_matrix <- confusionMatrix(xgb_predictions, test_dataData$INJURED_TYPE)
print(conf_matrix)


importance_matrix <- xgb.importance(feature_names = colnames(train_matrix), model = xgb_model)
print(importance_matrix)

# Plot feature importance
xgb.plot.importance(importance_matrix)








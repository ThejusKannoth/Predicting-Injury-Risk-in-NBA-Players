library(themis)
library(xgboost)
library(caret)
library(Matrix)
library(GGally)
library(corrplot)
library(ggplot2)
library(reshape2)
setwd('C:/Users/matts/Downloads')

df = read.csv('nba_injured_players.csv', sep=',')

data <- df[, -c(1,2,3,4,20,28,29,30)]

data$INJURED_TYPE <- as.factor(data$INJURED_TYPE)


#SMOTE on Train Data
train_data_balanced <- recipe(INJURED_TYPE ~ ., data = data) %>%
  step_smote(INJURED_TYPE, over_ratio = 0.5) %>%
  prep() %>%
  juice()

ctrl <- trainControl(method = "cv",         
                     number = 5,           
                     sampling = "smote") 

table(train_data_balanced$INJURED_TYPE)

trainIndex <- createDataPartition(train_data_balanced$INJURED_TYPE, p = 0.8, list = FALSE)

test_data <- data[-trainIndex, ]

train_matrix <- sparse.model.matrix(INJURED_TYPE ~ . -1, data = train_data_balanced)
test_matrix  <- sparse.model.matrix(INJURED_TYPE ~ . -1, data = test_data)

train_labels <- as.numeric(train_data_balanced$INJURED_TYPE) - 1
test_labels  <- as.numeric(test_data$INJURED_TYPE) - 1

dtrain <- xgb.DMatrix(data = train_matrix, label = train_labels)
dtest  <- xgb.DMatrix(data = test_matrix, label = test_labels)

# Define hyperparameters
xgb_params <- list(
  objective = "multi:softmax",
  num_class = length(unique(train_labels)),
  eval_metric = "mlogloss",
  eta = 0.1,
  max_depth = 6,  
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Train XGBoost model
xgb_model <- xgb.train(
  params = xgb_params,
  data = dtrain,
  trControl = ctrl,
  nrounds = 100,  
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 10,
  print_every_n = 10
)

# Predict on test data
xgb_predictions <- predict(xgb_model, newdata = dtest)
xgb_predictions <- factor(xgb_predictions, levels = 0:(length(unique(train_labels)) - 1), labels = levels(train_data_balanced$INJURED_TYPE))


# Check accuracy
conf_matrix <- confusionMatrix(xgb_predictions, test_data$INJURED_TYPE)
print(conf_matrix)

#Variable importance
importance_matrix <- xgb.importance(feature_names = colnames(train_matrix), model = xgb_model)
print(importance_matrix)

# Plot feature importance
xgb.plot.importance(importance_matrix)







###############################################################################
# Correlation Plot
################################################################################



ggpairs(data)

corData <- data

corData$INJURED_TYPE <- as.numeric(as.factor(data$INJURED_TYPE))


cor_matrix <- cor(corData, use = "complete.obs")  
print(cor_matrix)



###############################################################################
# Confustion Matrix Heat Map
################################################################################

cm_table <- as.data.frame(conf_matrix$table)
colnames(cm_table) <- c("Actual", "Predicted", "Freq")

ggplot(cm_table, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "black", size = 5) +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Confusion Matrix Heatmap") +
  theme_minimal()



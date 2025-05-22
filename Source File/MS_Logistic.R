library(nnet)
library(caret)
setwd('C:/Users/matts/Downloads')
df = read.csv('nba_injured_players.csv', sep=',')

###############################################################################

injury_data <- df[, -c(1,2,3,4,20,28,29,30)]

injury_data$INJURED_TYPE <- as.factor(injury_data$INJURED_TYPE)

###############################################################################

trainIndex <- createDataPartition(injury_data$INJURED_TYPE, p = 0.8, list = FALSE)

train_data_balanced <- recipe(INJURED_TYPE ~ ., data = injury_data) %>%
  step_smote(INJURED_TYPE, over_ratio = 0.5) %>%
  prep() %>%
  juice()

test_data <- injury_data[-trainIndex, ]

model <- multinom(INJURED_TYPE ~  . -1, data = train_data_balanced)

# Summary of the model
summary(model)


predictions <- predict(model, newdata = test_data)

conf_matrix <- confusionMatrix(predictions, test_data$INJURED_TYPE)
print(conf_matrix)

exp(coef(model))
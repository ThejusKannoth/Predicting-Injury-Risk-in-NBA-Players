library(ggplot2)
setwd('C:/Users/matts/Downloads')
df = read.csv('nba_injured_players.csv', sep=',')

data <- df[, -c(1,2,3,4,20,28,29,30,32)]

df$injury_numeric <- as.integer(factor(df$INJURED_TYPE))

################################################################################

data_scaled <- scale(data)

pca_result <- prcomp(data_scaled, center = TRUE, scale. = TRUE)

summary(pca_result)

print(pca_result$rotation)

head(pca_result$x)

screeplot(pca_result, main = "Scree Plot")

biplot(pca_result, main = "PCA Biplot")

################################################################################

pca_data2 <- as.data.frame(pca_result$x)

pca_data2$injury_numeric <- df$injury_numeric

pca_data_subset2 <- pca_data2[, c("PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "injury_numeric")]

model2 <- lm(injury_numeric ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6, data = pca_data_subset2)

summary(model2)

predictions2 <- predict(model2, newdata = pca_data_subset2)

results2 <- data.frame(Actual = pca_data_subset2$injury_numeric, Predicted = ceiling(predictions2))
print(results2)

results2$same <- results2$Actual == results2$Predicted
table(results2)
sum(results2$same == TRUE)


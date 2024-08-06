# Load necessary libraries
library(data.table)  # for fread and data manipulation
library(corrplot)    # for visualizing correlation matrices
library(dplyr)       # for data manipulation
library(ggplot2)    # for plotting

# Read the data
df1 <- fread('data/cleaned.csv')
# Correlation analysis
# Compute the correlation matrix
correlation_matrix <- cor(df1)

# Isolate the 'popularity' correlations and drop the self-correlation
popularity_correlation <- correlation_matrix['popularity', ]
popularity_correlation <- popularity_correlation[-which(names(popularity_correlation) == 'popularity')]
sorted_correlation <- sort(popularity_correlation, decreasing = TRUE)

# Verify the content of sorted_correlation
print(sorted_correlation)

# Loop through sorted_correlation
for (feature in names(sorted_correlation)) {
  if (abs(sorted_correlation[feature]) >= 0.01) {
    print(feature)
  }
}
# Variance threshold
# Drop the 'popularity' column to focus on features
df_feature <- df1[, setdiff(names(df1), 'popularity'), with=FALSE]

# Sequence of threshold values
threshold_values <- seq(0, 1, length.out = 11)

# Prepare lists to store the results
thresholds <- vector("list", length(threshold_values))
num_low_variance_cols <- vector("numeric", length(threshold_values))
low_variance_cols_list <- vector("list", length(threshold_values))

# Function to calculate variance and apply threshold
apply_variance_threshold <- function(df, threshold) {
  variances <- apply(df, 2, var) # Compute variance for each column
  low_variance_cols <- names(variances[variances < threshold])
  return(list("threshold" = threshold,
              "num_low_variance" = length(low_variance_cols),
              "low_variance_cols" = low_variance_cols))
}

# Loop over threshold values
results <- lapply(threshold_values, function(threshold) {
  apply_variance_threshold(df_feature, threshold)
})

# Extract data for plotting
thresholds <- sapply(results, function(x) x$threshold)
num_low_variance_cols <- sapply(results, function(x) x$num_low_variance)

# Plot results
plot_data <- data.frame(Thresholds = thresholds, NumLowVarianceCols = num_low_variance_cols)
ggplot(plot_data, aes(x = Thresholds, y = NumLowVarianceCols)) +
  geom_line(group = 1, colour = "blue") +
  geom_point(colour = "red") +
  ggtitle('Number of Low Variance Columns vs. Variance Threshold') +
  xlab('Variance Threshold') +
  ylab('Number of Low Variance Columns') +
  theme_minimal() +
  theme(panel.grid.major = element_line(colour = "gray", linetype = "dashed"))
# Assuming 'results' contains the results from the variance thresholding as previously defined

# Display the low variance columns for each threshold
for (i in seq_along(results)) {
  cat(sprintf("Variance Threshold: %.1f -> Low Variance Columns: %s\n",
              results[[i]]$threshold, toString(results[[i]]$low_variance_cols)))
}

# According to the plot, we would like to choose the features with threshold as 0.4 - 0.7 as the chosen features
# Find the indices for thresholds 0.4 to 0.7
selected_indices <- which(sapply(results, function(x) x$threshold) %in% seq(0.4, 0.7, by=0.1))

# Combine and select unique low variance columns from these indices
selected_features <- unique(unlist(lapply(results[selected_indices], function(x) x$low_variance_cols)))

# Print selected features
print("Variance Threshold selected features: ")
print(selected_features)

# PCA
library(factoextra)   # for PCA analysis
data <- fread('data/cleaned.csv')
# Remove the 'bathrooms_text' column
data <- data[, setdiff(names(data), 'bathrooms_text'), with=FALSE]

# Separate features and target
X <- data[, setdiff(names(data), 'popularity'), with=FALSE]
y <- data$popularity

# Standardize the features
X_scaled <- scale(X)

# Apply PCA to determine the optimal number of components
pca <- prcomp(X_scaled, center = TRUE, scale. = TRUE)
explained_variance_ratio <- pca$sdev^2 / sum(pca$sdev^2)
cumulative_explained_variance <- cumsum(explained_variance_ratio)

# Find the number of components for 95% variance
optimal_components <- which(cumulative_explained_variance >= 0.95)[1]

# Plot the explained variance to visualize the trade-off
ggplot(data.frame(Components = 1:length(cumulative_explained_variance), CumulativeVariance = cumulative_explained_variance), aes(x = Components, y = CumulativeVariance)) +
  geom_line() +
  geom_hline(yintercept = 0.95, linetype = "dashed", color = "red") +
  geom_vline(xintercept = optimal_components, linetype = "dashed", color = "red") +
  labs(title = "Explained Variance by Components", x = "Number of Components", y = "Cumulative Explained Variance") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_text(aes(x = optimal_components, y = 0.95, label = sprintf("95%% with %d components", optimal_components), vjust = -1))

# Apply PCA with the optimal number of components
pca_optimal <- prcomp(X_scaled, center = TRUE, scale. = TRUE, rank. = optimal_components)

# Get the loadings of the PCA components
loadings <- pca_optimal$rotation

# Calculate the absolute sum of loadings for each feature to see their overall contribution
feature_importance <- sort(colSums(abs(loadings)), decreasing = TRUE)
pca_features <- names(feature_importance)[1:23]

# Display the feature importance according to the PCA
print("PCA selected features: ")
print(pca_features)
# Chi-square
library(caret)

# Assuming 'X_scaled' and 'y' are already prepared
preProcValues <- preProcess(X_scaled, method = c("range"))  # Scale features between 0 and 1
X_scaled_non_negative <- predict(preProcValues, X_scaled)

# Set up control using train function with repeated cross-validation
fitControl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  savePredictions = "final",
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)


featureScore <- function(X, y) {
  scores <- sapply(X, function(x) {
    abs(cor(x, as.numeric(y)))  
  })
  names(scores) <- colnames(X)
  scores
}

topFeatures <- function(features, k) {
  names(sort(features, decreasing = TRUE))[1:k]
}

scores <- featureScore(X_scaled_non_negative, y)
selectedFeatures <- topFeatures(scores, k = 19)

print("Selected features based on custom scores:")
print(selectedFeatures)

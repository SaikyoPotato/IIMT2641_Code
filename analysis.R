options(repos = c(CRAN = "https://cran.rstudio.com/"))
required_packages <- c("tidyverse", "lubridate", "ggplot2", "corrplot", "forecast", 
                       "tsibble", "feasts", "ggfortify", "ggcorrplot", "caTools", 
                       "caret", "ROCR", "pROC", "e1071", "zoo", "GGally", "MASS")
installed_packages <- installed.packages()
for(p in required_packages){
  if(!(p %in% rownames(installed_packages))){
    install.packages(p)
  }
}

library(tidyverse)
library(lubridate)
library(ggplot2)
library(ggfortify)
library(ggcorrplot)
library(corrplot)
library(forecast)
library(tsibble)
library(feasts)
library(caTools)
library(caret)
library(ROCR)
library(pROC)


# Define file paths (adjust paths as necessary)
accident_file <- "./csv/accident_by_month_data.csv"
cpi_file <- "./csv/cpi_data_monthly.csv"
rainfall_file <- "./csv/hko_rf_monthly.csv"
hsi_file <- "./csv/hsi_index_data.csv"
traffic_file <- "./csv/traffic_data_monthly.csv"

# Load datasets
accidents <- read.csv(accident_file, stringsAsFactors = FALSE)
cpi_data <- read.csv(cpi_file, stringsAsFactors = FALSE)
rainfall <- read.csv(rainfall_file, stringsAsFactors = FALSE)
hsi_data <- read.csv(hsi_file, stringsAsFactors = FALSE)
traffic_data <- read.csv(traffic_file, stringsAsFactors = FALSE)

# Inspect datasets
head(accidents)
head(cpi_data)
head(rainfall)
head(hsi_data)
head(traffic_data)

# Convert year_month to Date format (first day of the month)
accidents$year_month <- as.Date(paste0(accidents$year_month, "-01"))
cpi_data$year_month <- as.Date(paste0(cpi_data$year_month, "-01"))
rainfall$year_month <- as.Date(paste0(rainfall$year_month, "-01"))
hsi_data$year_month <- as.Date(paste0(hsi_data$year_month, "-01"))
traffic_data$year_month <- as.Date(paste0(traffic_data$year_month, "-01"))

# Merge datasets
data_merged <- accidents %>%
  left_join(cpi_data, by = "year_month") %>%
  left_join(rainfall, by = "year_month") %>%
  left_join(hsi_data, by = "year_month") %>%
  left_join(traffic_data, by = "year_month")

# Check for missing values
summary(data_merged)

# Calculate accidents per thousand vehicles
data_merged <- data_merged %>%
  mutate(accidents_per_thousand = total_road_traffic_accidents / (total_in_thousands))

# Calculate traffic density (vehicles per day)
data_merged <- data_merged %>%
  mutate(traffic_density = total_in_thousands / no_of_days_in_the_year_month)

# Inspect the new features
head(data_merged)

# Select numerical columns for correlation
numeric_vars <- data_merged %>%
  select(total_road_traffic_accidents, composite_consumer_price_index, 
         year_on_year_percent_change, average_rainfall, open, high, low, close, 
         total_in_thousands, no_of_days_in_the_year_month, accidents_per_thousand, 
         traffic_density)

# Compute correlation matrix
cor_matrix <- cor(numeric_vars, use = "complete.obs")

ggcorrplot(cor_matrix, 
           lab = TRUE, 
           lab_size = 3, 
           colors = c("blue", "white", "red"), 
           title = "Correlation Matrix Heatmap") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Remove rows with missing or non-finite values
data_merged <- data_merged %>%
  filter(!is.na(total_road_traffic_accidents) & !is.na(average_rainfall) & !is.na(traffic_density))

# Scatter plot: Accidents vs. Average Rainfall
ggplot(data_merged, aes(x = average_rainfall, y = total_road_traffic_accidents)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(title = "Accidents vs. Average Rainfall", x = "Average Rainfall (mm)", y = "Total Road Traffic Accidents")

# Scatter plot: Accidents vs. Traffic Density
ggplot(data_merged, aes(x = traffic_density, y = total_road_traffic_accidents)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(title = "Accidents vs. Traffic Density", x = "Traffic Density (vehicles per day)", y = "Total Road Traffic Accidents")

# Check the length of the time series
if (nrow(data_merged) >= 24) {  # Ensure at least 2 years of monthly data
  # Convert to time series object
  accidents_ts <- ts(data_merged$total_road_traffic_accidents, start = c(year(min(data_merged$year_month)), month(min(data_merged$year_month))), frequency = 12)

  # Decompose the time series
  decomp <- decompose(accidents_ts, type = "additive")

  # Plot the decomposed components
  plot(decomp)
} else {
  print("Not enough data for time series decomposition")
}

# Plot total accidents over time
ggplot(data_merged, aes(x = year_month, y = total_road_traffic_accidents)) +
  geom_line(color = "blue", size = 1) +
  labs(title = "Traffic Accidents Over Time",
       x = "Year-Month", y = "Total Road Traffic Accidents") +
  theme_minimal()

  # Extract month as a separate column
data_merged <- data_merged %>%
  mutate(month = month(year_month, label = TRUE, abbr = TRUE))  # Extract month as a factor (Jan, Feb, etc.)

# Calculate average accidents by month
monthly_seasonality <- data_merged %>%
  group_by(month) %>%
  summarise(avg_accidents = mean(total_road_traffic_accidents, na.rm = TRUE))

# Plot average accidents by month
ggplot(monthly_seasonality, aes(x = month, y = avg_accidents)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Average Traffic Accidents by Month",
       x = "Month", y = "Average Accidents") +
  theme_minimal()


# Calculate 3-month rolling average
data_merged <- data_merged %>%
  arrange(year_month) %>%
  mutate(rolling_avg = zoo::rollmean(total_road_traffic_accidents, k = 3, fill = NA, align = "right"))

# Plot original data and rolling average
ggplot(data_merged, aes(x = year_month)) +
  geom_line(aes(y = total_road_traffic_accidents), color = "blue", size = 1, alpha = 0.6) +
  geom_line(aes(y = rolling_avg), color = "red", size = 1) +
  labs(title = "Traffic Accidents with 3-Month Rolling Average",
       x = "Year-Month", y = "Total Road Traffic Accidents") +
  theme_minimal()

# Plot with Loess smoothing
ggplot(data_merged, aes(x = year_month, y = total_road_traffic_accidents)) +
  geom_point(color = "blue", alpha = 0.5) +
  geom_smooth(method = "loess", span = 0.3, color = "red", size = 1) +
  labs(title = "Loess Smoothing of Traffic Accidents Over Time",
       x = "Year-Month", y = "Total Road Traffic Accidents") +
  theme_minimal()


  # Box plot to identify outliers
ggplot(data_merged, aes(x = month, y = total_road_traffic_accidents)) +
  geom_boxplot(fill = "lightblue", outlier.color = "red", outlier.size = 2) +
  labs(title = "Distribution of Traffic Accidents by Month",
       x = "Month", y = "Total Road Traffic Accidents") +
  theme_minimal()

  # Aggregate accidents by quarter
data_quarterly <- data_merged %>%
  mutate(quarter = yearquarter(year_month)) %>%
  group_by(quarter) %>%
  summarise(total_accidents = sum(total_road_traffic_accidents, na.rm = TRUE))

# Plot quarterly trends
ggplot(data_quarterly, aes(x = quarter, y = total_accidents)) +
  geom_line(color = "blue", size = 1) +
  labs(title = "Quarterly Traffic Accidents",
       x = "Quarter", y = "Total Accidents") +
  theme_minimal()


# Add time-based features
data_merged <- data_merged %>%
  mutate(year = year(year_month),        # Extract year
         month = month(year_month),      # Extract month as numeric
         quarter = quarter(year_month))  # Extract quarter

# Inspect the new features
head(data_merged)


# PART2: Principle Component Analysis

# Select relevant numerical variables for PCA
pca_vars <- data_merged %>%
  select(composite_consumer_price_index, year_on_year_percent_change, 
         consumer_price_index_a, year_on_year_percent_change_a, 
         consumer_price_index_b, year_on_year_percent_change_b, 
         consumer_price_index_c, year_on_year_percent_change_c, 
         average_rainfall, open, high, low, close, 
         total_in_thousands, no_of_days_in_the_year_month, 
         accidents_per_thousand, traffic_density)

# Handle missing values (e.g., remove rows with NA)
pca_vars_clean <- pca_vars %>%
  drop_na()

# Scale the data
pca_scaled <- scale(pca_vars_clean)

# Perform PCA
pca_result <- prcomp(pca_scaled, center = TRUE, scale. = TRUE)

# Summary of PCA
summary(pca_result)

# Scree plot
screeplot(pca_result, type = "lines", main = "Scree Plot")

# Scree plot using ggplot2
pca_var <- pca_result$sdev^2
pca_var_percent <- pca_var / sum(pca_var) * 100
pca_df <- data.frame(PC = paste0("PC", 1:length(pca_var_percent)),
                     Variance = pca_var_percent)

ggplot(pca_df, aes(x = PC, y = Variance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_line(aes(x = as.numeric(sub("PC", "", PC)), y = cumsum(Variance)), 
            group = 1, color = "red", size = 1) +
  geom_point(aes(x = as.numeric(sub("PC", "", PC)), y = cumsum(Variance)), 
             color = "red") +
  labs(title = "Scree Plot", x = "Principal Components", y = "Variance Explained (%)") +
  theme_minimal()

# Biplot of the first two principal components

autoplot(pca_result, data = data_merged, colour = 'total_road_traffic_accidents') +
  ggtitle("PCA Biplot") +
  theme_minimal()

  # Loadings of the first few principal components
pca_loadings <- pca_result$rotation[, 1:5]
print(pca_loadings)

# Cumulative variance explained
cum_var <- cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2)

# Find the number of components to explain at least 90% variance
num_pcs <- which(cum_var >= 0.90)[1]
cat("Number of principal components to retain for 90% variance:", num_pcs, "\n")


# Extract the selected principal components
selected_pcs <- as.data.frame(pca_result$x[, 1:num_pcs])

# Combine with the dependent variable
regression_data <- data_merged %>%
  select(year_month, total_road_traffic_accidents) %>%
  bind_cols(selected_pcs)

# Inspect the regression dataset
head(regression_data)

# Remove 'accidents_per_thousand' due to high correlation
data_final <- data_merged %>%
  select(-accidents_per_thousand)

data_final <- data_final %>%
  mutate(
    season = case_when(
      month %in% c("Dec", "Jan", "Feb") ~ "Winter",
      month %in% c("Mar", "Apr", "May") ~ "Spring",
      month %in% c("Jun", "Jul", "Aug") ~ "Summer",
      month %in% c("Sep", "Oct", "Nov") ~ "Autumn",
      TRUE ~ NA_character_  
    )
  )

# Define factor levels before splitting to prevent mismatches
data_final <- data_final %>%
  mutate(
    month = factor(month, levels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")),
    season = factor(season, levels = c("Winter", "Spring", "Summer", "Autumn"))
  )

# Re-split the data after defining factor levels
set.seed(2641)
split <- sample.split(data_final$total_road_traffic_accidents, SplitRatio = 0.7)
train_data <- subset(data_final, split == TRUE)
test_data <- subset(data_final, split == FALSE)
# Check the number of observations in training data
cat("Number of training observations:", nrow(train_data), "\n")


# Build the linear regression model with reduced predictors
multivar_lm <- lm(total_road_traffic_accidents ~ composite_consumer_price_index +
                   year_on_year_percent_change +
                   average_rainfall +
                   traffic_density,
                 data = train_data)

# Summary of the model
summary(multivar_lm)


# Predict on the test set
test_data$predicted_accidents <- predict(multivar_lm, newdata = test_data)

# Calculate evaluation metrics
rmse <- sqrt(mean((test_data$total_road_traffic_accidents - test_data$predicted_accidents)^2))
r_squared <- summary(multivar_lm)$r.squared

cat("RMSE:", rmse, "\n")
cat("R-squared:", r_squared, "\n")

# Plot Actual vs. Predicted
ggplot(test_data, aes(x = total_road_traffic_accidents, y = predicted_accidents)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, color = "red") +
  labs(title = "Actual vs. Predicted Traffic Accidents",
       x = "Actual Accidents",
       y = "Predicted Accidents") +
  theme_minimal()

# Calculate the 75th percentile
accident_threshold <- quantile(data_final$total_road_traffic_accidents, 0.75, na.rm = TRUE)

# Create a binary target variable
data_final <- data_final %>%
  mutate(high_risk = ifelse(total_road_traffic_accidents > accident_threshold, 1, 0))

# Split the data for logistic regression
set.seed(2641)
split_log <- sample.split(data_final$high_risk, SplitRatio = 0.7)
train_data_log <- subset(data_final, split_log == TRUE)
test_data_log <- subset(data_final, split_log == FALSE)

# Build the logistic regression model with reduced predictors
logistic_model <- glm(high_risk ~ composite_consumer_price_index +
                        year_on_year_percent_change +
                        average_rainfall +
                        traffic_density,
                      data = train_data_log,
                      family = binomial)

# Summary of the model
summary(logistic_model)


# Predict probabilities on the test set
test_data_log$predicted_prob <- predict(logistic_model, newdata = test_data_log, type = "response")

# Convert probabilities to binary outcomes (threshold = 0.5)
test_data_log$predicted_class <- ifelse(test_data_log$predicted_prob > 0.5, 1, 0)

# Confusion Matrix
conf_matrix <- table(Predicted = test_data_log$predicted_class, Actual = test_data_log$high_risk)
print(conf_matrix)

# Calculate Accuracy, Precision, Recall
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
precision <- ifelse(conf_matrix["1", "1"] + conf_matrix["1", "0"] > 0, 
                   conf_matrix["1", "1"] / sum(conf_matrix["1", ]), 0)
recall <- ifelse(conf_matrix["1", "1"] + conf_matrix["0", "1"] > 0, 
                conf_matrix["1", "1"] / sum(conf_matrix[, "1"]), 0)

cat("Accuracy:", round(accuracy, 2), "\n")
cat("Precision:", round(precision, 2), "\n")
cat("Recall:", round(recall, 2), "\n")

# Calculate AUC-ROC
roc_obj <- roc(test_data_log$high_risk, test_data_log$predicted_prob)
auc_value <- auc(roc_obj)
cat("AUC-ROC:", round(auc_value, 2), "\n")

# Plot ROC Curve
plot(roc_obj, main = "ROC Curve for Logistic Regression")
legend("bottomright", legend = paste("AUC =", round(auc_value, 2)), 
       col = "black", lwd = 2)

# Calculate Odds Ratios
odds_ratios <- exp(coef(logistic_model))
print(odds_ratios)

# Define the number of officers per accident
officers_per_accident <- 0.5

# Function to estimate required officers
estimate_officers <- function(predicted_accidents) {
  return(predicted_accidents * officers_per_accident)
}

# Define scenarios for rainfall increase
rainfall_increase_percent <- c(0, 10, 20, 30)  # 0%, 10%, 20%, 30%

# Create a data frame for scenarios
scenarios <- data.frame(increase_pct = rainfall_increase_percent)

# Calculate base rainfall
base_rainfall <- mean(train_data$average_rainfall, na.rm = TRUE)

# Function to simulate predictions with increased rainfall
simulate_scenario <- function(increase_pct, model, data) {
  # Increase rainfall
  new_rainfall <- base_rainfall * (1 + increase_pct / 100)
  
  # Create a copy of the data
  data_new <- data
  
  # Update average_rainfall
  data_new$average_rainfall <- new_rainfall
  
  # Predict accidents
  predicted_accidents <- predict(model, newdata = data_new, type = "response")
  
  return(mean(predicted_accidents, na.rm = TRUE))
}

# Apply simulation
scenarios$predicted_accidents <- sapply(scenarios$increase_pct, 
                                       simulate_scenario, 
                                       model = multivar_lm, 
                                       data = test_data)

# Estimate required officers
scenarios$required_officers <- estimate_officers(scenarios$predicted_accidents)

# View scenarios
print(scenarios)

# Plot required officers vs. rainfall increase
ggplot(scenarios, aes(x = increase_pct, y = required_officers)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Estimated Police Officers Required vs. Rainfall Increase",
       x = "Rainfall Increase (%)",
       y = "Estimated Required Officers") +
  theme_minimal()

 # Recalculate RMSE and R-squared for clarity
rmse <- sqrt(mean((test_data$total_road_traffic_accidents - test_data$predicted_accidents)^2))
r_squared <- summary(multivar_lm)$r.squared

cat("Multivariable Regression RMSE:", round(rmse, 2), "\n")
cat("Multivariable Regression R-squared:", round(r_squared, 2), "\n")

# Plot Actual vs. Predicted Accidents
ggplot(test_data, aes(x = total_road_traffic_accidents, y = predicted_accidents)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, color = "red") +
  labs(title = "Actual vs. Predicted Traffic Accidents",
       x = "Actual Accidents",
       y = "Predicted Accidents") +
  theme_minimal()

# Accuracy, Precision, Recall already calculated earlier

# Additionally, plot the ROC Curve with AUC
roc_obj <- roc(test_data_log$high_risk, test_data_log$predicted_prob)
plot(roc_obj, main = "ROC Curve for Logistic Regression Model")
legend("bottomright", legend = paste("AUC =", round(auc_value, 2)), 
       col = "black", lwd = 2)

# Compare RMSE and R-squared for regression
cat("Multivariable Regression - RMSE:", round(rmse, 2), " | R-squared:", round(r_squared, 2), "\n")

# AUC-ROC for logistic regression
cat("Logistic Regression - AUC-ROC:", round(auc_value, 2), "\n")
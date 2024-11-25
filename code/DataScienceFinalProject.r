# Data Science Final Project

# Names: Catalina Becu, Jorn Diesveld, Rebecca Hess, and Jorge Paredes

# Date: November - 2024

# [0] Preamble ----
set.seed(1)
library(tidyverse)
library(forcats)
library(farff)
library(pheatmap)
library(class)
library(caret)
library(randomForest)
library(xgboost)
library(glmnet)

# [1] Data ----
## Reading, selecting variables, and making numerical some categorical variables.
data <- readARFF("BSE/first quarter/datascience/final project 2/dataset_31_credit-g.arff") %>% 
    select(age, personal_status, job, housing, savings_status, checking_status, credit_amount, duration, purpose, credit_history, property_magnitude, housing, existing_credits, num_dependents, foreign_worker, installment_commitment, residence_since, class) %>% 
    separate(personal_status, into = c("sex", "p_status"), sep = " ") %>%
    mutate(class = ifelse(class == "good", 0, 1)) %>% 
    mutate(job = case_when(
        job == "unemp/unskilled non res" ~ 0,
        job == "unskilled resident" ~ 1,
        job == "skilled" ~ 2,
        job == "high qualif/self emp/mgmt" ~ 3,
        TRUE ~ NA)) %>% 
    mutate(savings_status = case_when(
        savings_status == "no known savings" ~ NA,
        savings_status == "<100" ~ "little",
        savings_status == "100<=X<500" ~ 'moderate',
        savings_status == "500<=X<1000" ~ 'quite rich',
        savings_status == ">=1000" ~ 'rich',
        TRUE ~ NA)) %>% 
    mutate(checking_status = case_when(
        checking_status == 'no checking' ~ NA,
        checking_status == "<0" ~ 'little',
        checking_status == "0<=X<200" ~ 'moderate',
        checking_status == ">=200" ~ 'rich',
        TRUE ~ NA)) %>% 
    rename(savings_account = savings_status, checking_account = checking_status) %>%
    mutate(checking_account = as.factor(checking_account), savings_account = as.factor(savings_account)) %>% 
    relocate(class)

# [2] Missing values / Factors treatment ----
df <- data

## Convert everything to numeric (most of them are factors)
label_encoders <- list()
factor_vars <- names(df)[sapply(df, function(x) is.factor(x) || is.character(x))]
for (column in factor_vars) {
  # Ensure the column is treated as a factor
  df[[column]] <- as.numeric(fct_inorder(as.factor(df[[column]]))) - 1
  label_encoders[[column]] <- levels(fct_inorder(as.factor(df[[column]])))
}
rm(column)
## Missing data treatment 
sapply(df, \(x) sum(is.na(x)))

df$savings_account <- ifelse(is.na(df$savings_account), -1, df$savings_account)
df$checking_account <- ifelse(is.na(df$checking_account), -1, df$checking_account)

# [3] Exploratory data analysis ----
## [3.1] Univariate analysis ----
data_long <- data %>% select(sex, age, credit_amount, duration) %>%
  pivot_longer(cols = c(age, credit_amount, duration), names_to = "variable", values_to = "value") %>% 
  mutate(variable = case_when(
    variable == "age" ~ "Age (in years)",
    variable == "credit_amount" ~ "Credit Amount (in DM)",
    variable == "duration" ~ "Duration (in months)"))

ggplot(data_long, aes(x = value, fill = sex)) +
  geom_histogram(aes(y = after_stat(density)), position = "identity", bins = 30, alpha = 0.5) +
  geom_density(aes(color = sex), linewidth = 1, fill = NA) +
  labs(y = "Density", x = " ", title = "Distribution by Sex", fill = "Sex:", color = "Sex:") +
  theme_light() +
  scale_x_continuous(n.breaks = 17) +
  scale_y_continuous(n.breaks = 10) +
  facet_wrap(~ variable, scales = "free") +
  scale_fill_manual(values = c("male" = "cornflowerblue", "female" = "firebrick")) +
  scale_color_manual(values = c("male" = "cornflowerblue", "female" = "firebrick")) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
        axis.title.y = element_text(size = 12),
        axis.title.x = element_text(size = 12), 
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        strip.text.x = element_text(face = "bold", color = "black", size = 12),
        legend.position = "bottom")

## [3.2] Credit amount ----
data %>% mutate(p_status = ifelse(p_status == "single"| p_status == "div/sep", "Single", "Married")) %>% 
ggplot(aes(x = credit_amount)) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "dodgerblue3", alpha = 0.7) +
    geom_density(color = "blue", linewidth = 1) +
    labs(y = "Density", x = "Credit Amount in DM", title = "Credit Amount Distribution by Status") +
    theme_light() +
    scale_x_continuous(n.breaks = 20) +
    scale_y_continuous(n.breaks = 10) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
          strip.text.x = element_text(face = "bold", color = "black", size = 12),
          axis.title.y = element_text(size = 12),
          axis.title.x = element_text(size = 12),
          axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
    facet_wrap(~ p_status, scales = "free")

data %>%
  mutate(p_status = ifelse(p_status == "single" | p_status == "div/sep", "Single", "Married")) %>%
  ggplot(aes(x = credit_amount, fill = p_status)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, alpha = 0.7, position = "identity") +
  geom_density(aes(color = p_status), linewidth = 1, fill = NA) +
  labs(
    y = "Density",
    x = "Credit Amount in DM",
    title = "Credit Amount Distribution by Marital Status",
    fill = "Marital Status:",
    color = "Marital Status:"
  ) +
  theme_light() +
  scale_x_continuous(n.breaks = 20) +
  scale_y_continuous(n.breaks = 10) +
  scale_fill_manual(values = c("Single" = "cornflowerblue", "Married" = "firebrick")) +
  scale_color_manual(values = c("Single" = "cornflowerblue", "Married" = "firebrick")) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
    axis.title.y = element_text(size = 12),
    axis.title.x = element_text(size = 12),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    strip.text.x = element_text(face = "bold", color = "black", size = 12),
    legend.position = "bottom"
  )

ggplot(data, aes(x = credit_amount)) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "dodgerblue3", alpha = 0.7) +
    geom_density(color = "blue", linewidth = 1) +
    labs(y = "Density", x = "Credit Amount in DM", title = "Credit Amount Distribution by product") +
    theme_light() +
    scale_x_continuous(n.breaks = 20) +
    scale_y_continuous(n.breaks = 10) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
          strip.text.x = element_text(face = "bold", color = "black", size = 12),
          axis.title.y = element_text(size = 12),
          axis.title.x = element_text(size = 12),
          axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
    facet_wrap(~ purpose, scales = "free")

data %>% mutate(class = ifelse(class == 0, "Good", "Bad")) %>%
ggplot(aes(x = credit_amount, fill = sex)) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, alpha = 0.7, position = "identity") +
    geom_density(aes(color = sex), linewidth = 1, fill = NA) +
    labs(
        y = "Density",
        x = "Credit Amount in DM",
        title = "Credit Amount Distribution by Class and Sex",
        fill = "Sex",
        color = "Sex"
    ) +
    theme_light() +
    scale_x_continuous(n.breaks = 20) +
    scale_y_continuous(n.breaks = 10) +
    theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
        strip.text.x = element_text(face = "bold", color = "black", size = 12),
        axis.title.y = element_text(size = 12),
        axis.title.x = element_text(size = 12),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        legend.position = "bottom") +
    facet_wrap(~ class, scales = "free") +
    scale_fill_manual(values = c("male" = "cornflowerblue", "female" = "firebrick")) +
    scale_color_manual(values = c("male" = "cornflowerblue", "female" = "firebrick"))

## [3.3] Duration ----
data %>% mutate(p_status = ifelse(p_status == "single"| p_status == "div/sep", "Single", "Married")) %>% 
ggplot(aes(x = duration)) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "dodgerblue3", alpha = 0.7) +
    geom_density(color = "blue", linewidth = 1) +
    labs(y = "Density", x = "Duration (in months)", title = "Credit Duration Distribution by Status") +
    theme_light() +
    scale_x_continuous(n.breaks = 20) +
    scale_y_continuous(n.breaks = 10) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
          strip.text.x = element_text(face = "bold", color = "black", size = 12),
          axis.title.y = element_text(size = 12),
          axis.title.x = element_text(size = 12),
          axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
    facet_wrap(~ p_status, scales = "free")

data %>%
  mutate(p_status = ifelse(p_status == "single" | p_status == "div/sep", "Single", "Married")) %>%
  ggplot(aes(x = duration, fill = p_status)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, alpha = 0.7, position = "identity") +
  geom_density(aes(color = p_status), linewidth = 1, fill = NA) +
  labs(
    y = "Density",
    x = "Duration (in months)",
    title = "Credit Duration Distribution by Marital Status",
    fill = "Marital Status:",
    color = "Marital Status:"
  ) +
  theme_light() +
  scale_x_continuous(n.breaks = 20) +
  scale_y_continuous(n.breaks = 10) +
  scale_fill_manual(values = c("Single" = "cornflowerblue", "Married" = "firebrick")) +
  scale_color_manual(values = c("Single" = "cornflowerblue", "Married" = "firebrick")) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
    axis.title.y = element_text(size = 12),
    axis.title.x = element_text(size = 12),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    strip.text.x = element_text(face = "bold", color = "black", size = 12),
    legend.position = "bottom"
  )

ggplot(data, aes(x = duration)) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "dodgerblue3", alpha = 0.7) +
    geom_density(color = "blue", linewidth = 1) +
    labs(y = "Density", x = "Duration (in months)", title = "Credit Duration Distribution by product") +
    theme_light() +
    scale_x_continuous(n.breaks = 20) +
    scale_y_continuous(n.breaks = 10) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
          strip.text.x = element_text(face = "bold", color = "black", size = 12),
          axis.title.y = element_text(size = 12),
          axis.title.x = element_text(size = 12),
          axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
    facet_wrap(~ purpose, scales = "free")

data %>% mutate(class = ifelse(class == 0, "Good", "Bad")) %>%
ggplot(aes(x = duration, fill = sex)) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, alpha = 0.7, position = "identity") +
    geom_density(aes(color = sex), linewidth = 1, fill = NA) +
    labs(
        y = "Density",
        x = "Duration (in months)",
        title = "Credit Duration Distribution by Class and Sex",
        fill = "Sex",
        color = "Sex"
    ) +
    theme_light() +
    scale_x_continuous(n.breaks = 20) +
    scale_y_continuous(n.breaks = 10) +
    theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
        strip.text.x = element_text(face = "bold", color = "black", size = 12),
        axis.title.y = element_text(size = 12),
        axis.title.x = element_text(size = 12),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
        legend.position = "bottom") +
    facet_wrap(~ class, scales = "free") +
    scale_fill_manual(values = c("male" = "cornflowerblue", "female" = "firebrick")) +
    scale_color_manual(values = c("male" = "cornflowerblue", "female" = "firebrick"))

## [3.4] Correlation matrix ----
pheatmap(cor(df, use = "complete.obs"), 
         display_numbers = TRUE, 
         number_color = "black",
         main = "Feature Correlation Heatmap", treeheight_row = F, treeheight_col = F)

# [4] Preprocessing ----
## Feature engeenering
### Add: squared age, squared credit amount, squared duration, and squared number of dependents
df <- df %>% mutate(age2 = age^2, credit_amount2 = credit_amount^2, duration2 = duration^2, num_dependents2 = num_dependents^2)

## Normalization across the numeric variables
num_vars <- c('duration', 'credit_amount', 'installment_commitment', 'residence_since', 'age', 'existing_credits', 'num_dependents', 'age2', 'credit_amount2', 'duration2', 'num_dependents2')
df <- df %>% mutate(across(all_of(num_vars), scale))

# [5] Models ----
## [5.0] Train-test split ----
train_index <- sample(1:nrow(df), 0.75 * nrow(df))
X <- df %>% select(-class)
y <- as.factor(df$class)

train_x <- X[train_index, ]
train_y <- y[train_index]
test_x <- X[-train_index, ]
test_y <- y[-train_index]
rm(train_index)

## [5.1] KNN ----
train_data <- cbind(train_x, class = train_y)
control <- trainControl(method = "cv", number = 7)
knn_model <- train(class ~ ., data = train_data, method = "knn", tuneGrid = data.frame(k = 1:25), trControl = control)

best_k <- knn_model$bestTune$k
knn_predictions <- knn(train = train_x, test = test_x, cl = train_y, k = best_k)
conf_matrix_knn <- confusionMatrix(knn_predictions, test_y)

## [5.2] Random Forest ----
ntree_grid  <- 100*c(1:5)
mtry_grid <- c(2, 3, floor(sqrt(ncol(train_data) -1)), 5)
rf_results <- list()

for(ntree in ntree_grid){
    rf_grid <- expand.grid(mtry = mtry_grid)
    rf_tuned <- train(
        class ~ ., 
        data = train_data, 
        method = "rf",
        trControl = control,
        tuneGrid = rf_grid, 
        ntree = ntree 
    )
    
    rf_results[[as.character(ntree)]] <- list(
        ntree = ntree,
        model = rf_tuned,
        best_mtry = rf_tuned$bestTune$mtry,
        accuracy = max(rf_tuned$results$Accuracy),
        results = rf_tuned$results
    )
}

# Find the best ntree and mtry combination based on accuracy
best_ntree <- names(which.max(sapply(rf_results, function(x) x$accuracy)))
best_result <- rf_results[[best_ntree]]

# Print the best parameters
cat("Best ntree:", best_result$ntree, "\n")
cat("Best mtry:", best_result$best_mtry, "\n")
cat("Best Accuracy:", best_result$accuracy, "\n")

rf_model <- randomForest(class ~ .,  data = train_data, ntree = best_result$ntree, mtry = best_result$best_mtry)
rf_predictions <- predict(rf_model, newdata = test_x)
conf_matrix_rf <- confusionMatrix(rf_predictions, test_y)

## [5.3] XGBoost ----
train_matrix <- model.matrix(class ~ ., data = train_data)[, -1]
test_matrix <- model.matrix(~ ., data = test_x)[, -1]
train_label <- train_y
test_label <- test_y

xgb_control <- trainControl(method = "cv", number = 7, verboseIter = FALSE)

xgb_model <- train(x = train_matrix, y = train_label, method = "xgbTree", trControl = xgb_control, 
    tuneGrid = expand.grid(nrounds = seq(50, 200, by = 50), max_depth = c(3, 6), eta = c(0.1), gamma = c(0, 0.1), colsample_bytree = 0.8, min_child_weight = c(1,3), subsample = 0.8))

xgb_predictions <- predict(xgb_model, newdata = test_matrix)
conf_matrix_xgb <- confusionMatrix(xgb_predictions, test_label)

## [5.4] Elastic net with logit ----
train_matrix <- model.matrix(class ~ ., data = train_data)[, -1]
test_matrix <- model.matrix(~ ., data = test_x)[, -1]

train_label <- as.numeric(as.character(train_y))
test_label <- as.numeric(as.character(test_y))

# Cross-validation to find the best alpha
models <- list()
results <- data.frame(alpha = numeric(), lambda = numeric(), accuracy = numeric())
for(i in 0:50){
    name <- paste0("alpha", i/50)
    model <- cv.glmnet(x = train_matrix, y = train_label, family = "binomial", alpha = i/50, lambda = NULL)
    models[[name]] <- model
    # Make predictions on the test set
    predictions <- predict(model, newx = test_matrix, s = "lambda.min", type = "class")
     # Evaluate accuracy
    conf_matrix <- confusionMatrix(as.factor(predictions), as.factor(test_label))
    accuracy <- conf_matrix$overall["Accuracy"]
    
    # Save results
    results <- rbind(results, data.frame(alpha = i/50, lambda = model$lambda.min, accuracy = accuracy))
}
results <- results %>% filter(alpha != 0 & alpha != 1) %>% arrange(desc(accuracy))
# pick the best alpha
alpha <- results[1, "alpha"]

# Ridge
ridge_model <- cv.glmnet(x = train_matrix, y = train_label, family = "binomial", alpha = 0, lambda = NULL)
ridge_predictions <- predict(ridge_model, newx = test_matrix, s = "lambda.min", type = "class")
conf_matrix_ridge <- confusionMatrix(as.factor(ridge_predictions), as.factor(test_label))

# Elastic net
elastic_net_model <- cv.glmnet(x = train_matrix, y = train_label, family = "binomial", alpha = alpha, lambda = NULL)
elastic_net_predictions <- predict(elastic_net_model, newx = test_matrix, s = "lambda.min", type = "class")
conf_matrix_elastic <- confusionMatrix(as.factor(elastic_net_predictions), as.factor(test_label))

# Lasso
lasso_model <- cv.glmnet(x = train_matrix, y = train_label, family = "binomial", alpha = 1, lambda = NULL)
lasso_predictions <- predict(lasso_model, newx = test_matrix, s = "lambda.min", type = "class")
conf_matrix_lasso <- confusionMatrix(as.factor(lasso_predictions), as.factor(test_label))

# [6] Results ----
### Get all the confusion matrices in a list for comparison
confusion_matrices <- list(knn = conf_matrix_knn, rf = conf_matrix_rf, xgb = conf_matrix_xgb, ridge = conf_matrix_ridge, elastic_net = conf_matrix_elastic, lasso = conf_matrix_lasso)

## Get the accuracy, and the CI into a dataframe
accuracy <- data.frame(model = names(confusion_matrices), 
                      accuracy = sapply(confusion_matrices, function(x) x$overall["Accuracy"]),
                      lower_ci = sapply(confusion_matrices, function(x) x$overall["AccuracyLower"]),
                      upper_ci = sapply(confusion_matrices, function(x) x$overall["AccuracyUpper"]))
rownames(accuracy) <- NULL
accuracy  <- accuracy %>% mutate(model = case_when(model == "knn" ~ "K-Nearest-Neighbors",
                                                   model == "rf" ~ "Random Forest",
                                                   model == "xgb" ~ "XGBoost",
                                                   model == "ridge" ~ "Ridge",
                                                   model == "elastic_net" ~ "Elastic Net",
                                                   model == "lasso" ~ "Lasso")) %>% 
            mutate(model = fct_relevel(model, "Lasso", "Elastic Net", "Ridge", "XGBoost", "Random Forest", "K-Nearest-Neighbors"))

# plot this results in a geom_point plot with error bars
ggplot(accuracy, aes(x = model, y = accuracy)) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), width = 0.2) +
    labs(y = "Accuracy", x = "Model", title = "Model Comparison") +
    theme_light() + coord_flip() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
          strip.text.x = element_text(face = "bold", color = "black", size = 12),
          axis.title.y = element_text(size = 12),
          axis.title.x = element_text(size = 12),
          axis.text.x = element_text(angle = 270, vjust = 0.5, hjust=1)) +
    scale_y_continuous(n.breaks = 20)

# End-of-File ----

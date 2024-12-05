# Data Science Final Project

# Names: Catalina Becu, Jorn Diesveld, Rebecca Hess, and Jorge Paredes

# Date: November - 2024

# [0] Preamble ----
set.seed(1)
library(tidyverse)
library(forcats)
library(pheatmap)
library(class)
library(caret)
library(randomForest)
library(xgboost)
library(glmnet)
library(pROC)
library(FNN)

# [1] Data ----
## Reading, selecting variables, and making numerical some categorical variables.

data <- read.csv('https://raw.githubusercontent.com/jparedes-m/DataScienceBSE/refs/heads/main/data/credit.csv') %>% 
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
        savings_status == "no known savings" ~ 'no known savings',
        savings_status == "<100" ~ "little",
        savings_status == "100<=X<500" ~ 'moderate',
        savings_status == "500<=X<1000" ~ 'quite rich',
        savings_status == ">=1000" ~ 'rich',
        TRUE ~ NA)) %>% 
    mutate(checking_status = case_when(
        checking_status == 'no checking' ~ "no checking",
        checking_status == "<0" ~ 'little',
        checking_status == "0<=X<200" ~ 'moderate',
        checking_status == ">=200" ~ 'rich',
        TRUE ~ NA)) %>% 
    rename(savings_account = savings_status, checking_account = checking_status) %>%
    mutate(checking_account = as.factor(checking_account), savings_account = as.factor(savings_account), property_magnitude = as.factor(property_magnitude)) %>% 
    relocate(class)

# [2] Missing values / Factors treatment ----
df <- data

## Missing data treatment 
sapply(df, \(x) 100*mean(is.na(x)))
mode_fctr <- function(x) levels(x)[which.max(tabulate(match(x, levels(x))))]

df <- mutate(df,
            savings_account = ifelse(is.na(savings_account), mode_fctr(savings_account), savings_account),
            property_magnitude = ifelse(is.na(property_magnitude), mode_fctr(property_magnitude), property_magnitude))

## Convert everything to numeric (most of them are factors)
label_encoders <- list()
factor_vars <- names(df)[sapply(df, function(x) is.factor(x) || is.character(x))]
for (column in factor_vars) {
  # Ensure the column is treated as a factor
  df[[column]] <- as.numeric(fct_inorder(as.factor(df[[column]]))) - 1
  label_encoders[[column]] <- levels(fct_inorder(as.factor(df[[column]])))
}
rm(column)
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

#ggsave("Aplicaciones/Overleaf/Foundations of Data Science - BSE Group/assets/figures/3_1_graph.png", width = 9, height = 10, bg = "white")

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
    facet_wrap(~ p_status, scales = "free", nrow = 2)

#ggsave("Aplicaciones/Overleaf/Foundations of Data Science - BSE Group/assets/figures/3_2_1_graph.png", width = 9, height = 10, bg = "white")

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

#ggsave("Aplicaciones/Overleaf/Foundations of Data Science - BSE Group/assets/figures/3_2_2_graph.png", width = 9, height = 10, bg = "white")

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
    facet_wrap(~ purpose, scales = "free", nrow = 2)

#ggsave("Aplicaciones/Overleaf/Foundations of Data Science - BSE Group/assets/figures/3_2_3_graph.png", width = 15, height = 10, bg = "white")

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
    facet_wrap(~ class, scales = "free", nrow = 2) +
    scale_fill_manual(values = c("male" = "cornflowerblue", "female" = "firebrick")) +
    scale_color_manual(values = c("male" = "cornflowerblue", "female" = "firebrick"))

#ggsave("Aplicaciones/Overleaf/Foundations of Data Science - BSE Group/assets/figures/3_2_4_graph.png", width = 9, height = 10, bg = "white")

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
    facet_wrap(~ p_status, scales = "free", nrow = 2)

#ggsave("Aplicaciones/Overleaf/Foundations of Data Science - BSE Group/assets/figures/3_3_1_graph.png", width = 9, height = 10, bg = "white")

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

#ggsave("Aplicaciones/Overleaf/Foundations of Data Science - BSE Group/assets/figures/3_3_2_graph.png", width = 9, height = 10, bg = "white")

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
    facet_wrap(~ purpose, scales = "free", nrow = 2)

#ggsave("Aplicaciones/Overleaf/Foundations of Data Science - BSE Group/assets/figures/3_3_3_graph.png", width = 15, height = 10, bg = "white")

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
    facet_wrap(~ class, scales = "free", nrow = 2) +
    scale_fill_manual(values = c("male" = "cornflowerblue", "female" = "firebrick")) +
    scale_color_manual(values = c("male" = "cornflowerblue", "female" = "firebrick"))

#ggsave("Aplicaciones/Overleaf/Foundations of Data Science - BSE Group/assets/figures/3_3_4_graph.png", width = 9, height = 10, bg = "white")

## [3.4] Correlation matrix ----
map <- pheatmap(cor(df, use = "complete.obs"), 
         display_numbers = TRUE, 
         number_color = "black",
         main = "Feature Correlation Heatmap", treeheight_row = F, treeheight_col = F)
save_pheatmap <- function(x, filename, width=12, height=12){
  stopifnot(!missing(x))
  stopifnot(!missing(filename))
  if(grepl(".png",filename)){
    png(filename, width=width, height=height, units = "in", res=300)
    grid::grid.newpage()
    grid::grid.draw(x$gtable)
    dev.off()
  }
  else if(grepl(".pdf",filename)){
    pdf(filename, width=width, height=height)
    grid::grid.newpage()
    grid::grid.draw(x$gtable)
    dev.off()
  }
  else{
    print("Filename did not contain '.png' or '.pdf'")
  }
}


#save_pheatmap(map, "Aplicaciones/Overleaf/Foundations of Data Science - BSE Group/assets/figures/3_4_graph.png", width = 10, height = 10)
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
y <- factor(df$class, levels = c(0,1), labels = c("good", "bad"))

train_x <- X[train_index, ]
train_y <- y[train_index]

test_x <- X[-train_index, ]
test_y <- y[-train_index]
rm(train_index)

## [5.1] KNN ----
train_data <- cbind(train_x, class = train_y)
control <- trainControl(method = "cv", number = 7, classProbs = TRUE)
knn_model <- train(class ~ ., data = train_data, method = "knn", tuneGrid = data.frame(k = 1:25), trControl = control, metric = 'ROC')

best_k <- knn_model$bestTune$k
# instead of hard classification we will use the class probabilities
knn_predictions <- predict(knn_model, newdata = test_x, type = "raw")
conf_matrix_knn <- confusionMatrix(knn_predictions, test_y)

knn_probabilities <- predict(knn_model, newdata = test_x, type = "prob")
# we are interested in predicting the 'bad' class
roc_knn <- roc(test_y, knn_probabilities[, "bad"])

## [5.2] Random Forest ----
ntree_grid  <- 100*c(1:5)
mtry_grid <- c(2, 3, floor(sqrt(ncol(train_data) -1)), 5)
rf_results <- list()
control <- trainControl(method = "cv", number = 7, classProbs = TRUE, summaryFunction = twoClassSummary)

for(ntree in ntree_grid){
    rf_grid <- expand.grid(mtry = mtry_grid)
    rf_tuned <- train(
        class ~ ., 
        data = train_data, 
        method = "rf",
        trControl = control,
        tuneGrid = rf_grid, 
        metric = 'ROC',
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

rf_probabilities <- predict(rf_model, newdata = test_x, type = "prob")

# compute the ROC of this model
roc_rf <- roc(test_y, rf_probabilities[, "bad"])

## [5.3] XGBoost ----
train_matrix <- model.matrix(class ~ ., data = train_data)[, -1]
test_matrix <- model.matrix(~ ., data = test_x)[, -1]
train_label <- train_y
test_label <- test_y

xgb_control <- trainControl(method = "cv", number = 7, verboseIter = FALSE, classProbs = TRUE, summaryFunction = twoClassSummary)

xgb_model <- train(x = train_matrix, y = train_label, method = "xgbTree", trControl = xgb_control, verbosity = 0, metric = 'ROC',
    tuneGrid = expand.grid(nrounds = seq(50, 200, by = 50), max_depth = c(3, 6), eta = c(0.1), gamma = c(0, 0.1), colsample_bytree = 0.8, min_child_weight = c(1,3), subsample = 0.8))

xgb_predictions <- predict(xgb_model, newdata = test_matrix)
conf_matrix_xgb <- confusionMatrix(xgb_predictions, test_label)

xgb_probabilities <- predict(xgb_model, newdata = test_matrix, type = "prob")

roc_xgb <- roc(test_label, xgb_probabilities[, "bad"])

## [5.4] Elastic net with logit ----
train_matrix <- model.matrix(class ~ ., data = train_data)[, -1]
test_matrix <- model.matrix(~ ., data = test_x)[, -1]

train_label <- as.numeric(train_y)-1
test_label <- as.numeric(test_y)-1

# Cross-validation to find the best alpha
models <- list()
results <- data.frame(alpha = numeric(), lambda = numeric(), auc = numeric())
for (i in 0:50) {
    name <- paste0("alpha", i / 50)
    model <- cv.glmnet(x = train_matrix, y = train_label, family = "binomial", alpha = i / 50, lambda = NULL)
    models[[name]] <- model
    
    # Predicciones como probabilidades
    probabilities <- predict(model, newx = test_matrix, s = "lambda.min", type = "response")
    
    # Calcular AUC
    auc <- suppressMessages(auc(test_label, probabilities))
    
    # Guardar resultados
    results <- rbind(results, data.frame(alpha = i / 50, lambda = model$lambda.min, auc = auc))
}

# Seleccionar el mejor α basado en AUC
results <- results %>% filter(alpha != 0 & alpha != 1) %>% arrange(desc(auc))
alpha <- results[1, "alpha"]

# Ridge
ridge_model <- cv.glmnet(x = train_matrix, y = train_label, family = "binomial", alpha = 0, lambda = NULL)
#png("/Users/rebeccahess/Documents/BSE Sem 1/datascience/ridge_graph.png", width = 800, height = 600) 
plot(ridge_model)
title("Ridge Coefficient Shrinkage Plot\n")
dev.off()   

# Print the coefficients for the Ridge model at lambda.1se
ridge_coefs_1se <- coef(ridge_model, s = "lambda.1se")
print(ridge_coefs_1se)
# Print the coefficients for the Ridge model at lambda.1se
ridge_coefs_min <- coef(ridge_model, s = "lambda.min")
print(ridge_coefs_min)

ridge_probabilities <- predict(ridge_model, newx = test_matrix, s = "lambda.min", type = "response")
ridge_predictions <- predict(ridge_model, newx = test_matrix, s = "lambda.min", type = "class")
ridge_predictions <- factor(ridge_predictions, levels = c(0, 1))
test_label_factor <- factor(test_label, levels = c(0, 1))

conf_matrix_ridge <- confusionMatrix(ridge_predictions, test_label_factor)

roc_ridge <- roc(test_label, ridge_probabilities)
# Generate the plot and add the title


# Elastic net
elastic_net_model <- cv.glmnet(x = train_matrix, y = train_label, family = "binomial", alpha = alpha, lambda = NULL)
#png("/Users/rebeccahess/Documents/BSE Sem 1/datascience/elastic_graph.png", width = 800, height = 600) 
plot(elastic_net_model)
title("Elastic Net Coefficient Shrinkage Plot\n")
dev.off()

# Print the coefficients for the Ridge model at lambda.1se
elastic_coefs_1se <- coef(elastic_net_model, s = "lambda.1se")
print(elastic_coefs_1se)
# Print the coefficients for the Ridge model at lambda.1se
elastic_coefs_min <- coef(elastic_net_model, s = "lambda.min")
print(elastic_coefs_min)


elastic_net_probabilities <- predict(elastic_net_model, newx = test_matrix, s = "lambda.min", type = "response")
elastic_net_predictions <- predict(elastic_net_model, newx = test_matrix, s = "lambda.min", type = "class")
elastic_net_predictions <- factor(elastic_net_predictions, levels = c(0, 1))
test_label_factor <- factor(test_label, levels = c(0, 1))
conf_matrix_elastic <- confusionMatrix(elastic_net_predictions, test_label_factor)

roc_elastic <- roc(test_label, as.vector(elastic_net_probabilities))

# Lasso
lasso_model <- cv.glmnet(x = train_matrix, y = train_label, family = "binomial", alpha = 1, lambda = NULL)

#png("/Users/rebeccahess/Documents/BSE Sem 1/datascience/lasso_graph.png", width = 800, height = 600) 
plot(lasso_model)
title("Lasso Coefficient Shrinkage Plot\n")

lasso_coefs_min <- coef(lasso_model, s = "lambda.min")
print(lasso_coefs_min)
lasso_coefs_1se <- coef(lasso_model, s = "lambda.1se")
print(lasso_coefs_1se)


lasso_probabilities <- predict(lasso_model, newx = test_matrix, s = "lambda.min", type = "response")
lasso_predictions <- predict(lasso_model, newx = test_matrix, s = "lambda.min", type = "class")
lasso_predictions <- factor(lasso_predictions, levels = c(0, 1))
test_label_factor <- factor(test_label, levels = c(0, 1))
conf_matrix_lasso <- confusionMatrix(lasso_predictions, test_label_factor)
roc_lasso <- roc(test_label, as.vector(lasso_probabilities))

# Logit without penalization
logit_model <- glm(class ~ ., data = train_data, family = "binomial")
logit_probabilities <- predict(logit_model, newdata = test_x, type = "response")
logit_predictions <- ifelse(logit_probabilities > 0.5, 1, 0)
logit_predictions <- factor(logit_predictions, levels = c(0, 1))
test_label <- factor(test_label, levels = c(0, 1))
conf_matrix_logit <- confusionMatrix(logit_predictions, test_label)
roc_logit <- roc(test_label, logit_probabilities)


# [6] Results ----
### Get all the confusion matrices in a list for comparison
confusion_matrices <- list(knn = conf_matrix_knn, rf = conf_matrix_rf, xgb = conf_matrix_xgb, ridge = conf_matrix_ridge, elastic_net = conf_matrix_elastic, lasso = conf_matrix_lasso, logit = conf_matrix_logit)

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
                                                   model == "lasso" ~ "Lasso", 
                                                   model == 'logit' ~ "Logit")) %>% 
            mutate(model = fct_relevel(model, "Logit", "Lasso", "Elastic Net", "Ridge", "XGBoost", "Random Forest", "K-Nearest-Neighbors"))

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

#png("/Users/rebeccahess/Documents/BSE Sem 1/datascience/roc_curves_plot.png", width = 1200, height = 800, res = 150)
# Plot all the roc curves with the AUC in the legend with the plot function
plot(y=roc_knn$sensitivities, x=1-roc_knn$specificities, col = "red", lwd = 2, type = "l", main = "ROC Curves", xlab = "False Positive Rate", ylab = "True Positive Rate")
lines(y=roc_rf$sensitivities, x=1-roc_rf$specificities, col = "blue", lwd = 2)
lines(y=roc_xgb$sensitivities, x=1-roc_xgb$specificities, col = "green", lwd = 2)
lines(y=roc_ridge$sensitivities, x=1-roc_ridge$specificities, col = "purple", lwd = 2)
lines(y=roc_elastic$sensitivities, x=1-roc_elastic$specificities, col = "orange", lwd = 2)
lines(y=roc_lasso$sensitivities, x=1-roc_lasso$specificities, col = "black", lwd = 2)
lines(y=roc_logit$sensitivities, x=1-roc_logit$specificities, col = "brown", lwd = 2)
legend("bottomright", 
       legend = c(
         paste0("K-Nearest-Neighbors (AUC = ", round(auc(roc_knn), 3), ")"),
         paste0("Random Forest (AUC = ", round(auc(roc_rf), 3), ")"),
         paste0("XGBoost (AUC = ", round(auc(roc_xgb), 3), ")"),
         paste0("Ridge (AUC = ", round(auc(roc_ridge), 3), ")"),
         paste0("Elastic Net (AUC = ", round(auc(roc_elastic), 3), ")"),
         paste0("Lasso (AUC = ", round(auc(roc_lasso), 3), ")"),
          paste0("Logit (AUC = ", round(auc(roc_logit), 3), ")")
       ),
       col = c("red", "blue", "green", "purple", "orange", "black", "brown"), 
       lwd = 2)
# End-of-File ----
dev.off()

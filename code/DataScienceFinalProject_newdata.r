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
    mutate(repayment_burden = credit_amount/duration) %>%
    rename(savings_account = savings_status, checking_account = checking_status) %>%
    mutate(checking_account = as.factor(checking_account), savings_account = as.factor(savings_account)) %>% 
    relocate(class)

# [2] Missing values / Factors treatment ----
df <- data

## Missing data treatment 
sapply(df, \(x) 100*mean(is.na(x)))
mode_fctr <- function(x) levels(x)[which.max(tabulate(match(x, levels(x))))]

df$savings_account <- ifelse(is.na(df$savings_account), mode_fctr(df$savings_account), df$savings_account)
df$checking_account <- ifelse(is.na(df$checking_account), mode_fctr(df$checking_account), df$checking_account)

## Convert everything to numeric (most of them are factors)
label_encoders <- list()
factor_vars <- names(df)[sapply(df, function(x) is.factor(x) || is.character(x))]
for (column in factor_vars) {
  # Ensure the column is treated as a factor
  df[[column]] <- as.numeric(fct_inorder(as.factor(df[[column]]))) - 1
  label_encoders[[column]] <- levels(fct_inorder(as.factor(df[[column]])))
}
rm(column)

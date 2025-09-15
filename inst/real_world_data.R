#' This script follows the model applications of Thielman et al. (2020)


# California Housing Data -------------------------------------------------
# The California Housing Data is provided within the library

data(ca_housing)

#reduced_df <- ca_housing[, c("MedInc", "Population", "target"), drop = FALSE]
#input_vars <- reduced_df[, setdiff(names(reduced_df), "target"), drop = FALSE]

input_vars <- ca_housing[, setdiff(names(ca_housing), "target"), drop = FALSE]

ca_housing_split <- random_split(input_vars, normalization=FALSE)

targets_ca_housing <- ca_housing$target

train_ca_housing <- ca_housing_split$train


# ---- Prepocessing ---- #
# features will be scaled to [-1, 1]
pm1 <- pm1_scaler(train_ca_housing)
pm1

train_ca_housing <- transform_pm1(train_ca_housing, pm1)
# sanity check
sapply(train_ca_housing, min)
sapply(train_ca_housing, max)


# Ziehen von val/test splits
val_ca_housing <- ca_housing_split$validation
val_targets_ca_housing <- targets_ca_housing[as.integer(rownames(val_ca_housing))]
# [-1, 1] Scaling
val_ca_housing <- transform_pm1(val_ca_housing, pm1, clip=TRUE) # clippen ja nein?
sapply(val_ca_housing, min)
sapply(val_ca_housing, max)

test_ca_housing <- ca_housing_split$test
test_targets_ca_housing <- targets_ca_housing[as.integer(rownames(test_ca_housing))]
# [-1, 1] Scaling
test_ca_housing <- transform_pm1(test_ca_housing, pm1, clip=TRUE)
sapply(test_ca_housing, min)
sapply(test_ca_housing, max)

# targets will be standard normalized
train_targets <- targets_ca_housing[as.integer(rownames(train_ca_housing))]

norm_targets <- normalize_targets(train_targets, val_targets_ca_housing, test_targets_ca_housing)
# reassign normalized targets
train_targets_ca_housing <- norm_targets$train
val_targets_ca_housing <- norm_targets$validation
test_targets_ca_housing <- norm_targets$test

# put them back into one vector by mapping them to their original indices
targets_std_full <- rep(NA_real_, length(targets_ca_housing))
targets_std_full[as.integer(rownames(train_ca_housing))] <- train_targets_ca_housing
targets_std_full[as.integer(rownames(val_ca_housing))]   <- val_targets_ca_housing
targets_std_full[as.integer(rownames(test_ca_housing))]  <- test_targets_ca_housing



# Model

n_inputs <- length(input_vars)

ca_housing_loader <- DataLoader(train_ca_housing, batch_size = 256)

nam_housing <- train_namls(ca_housing_loader, targets_std_full, length(input_vars),  c(32, 64, 128, 256), t(val_ca_housing), val_targets_ca_housing,
                             optimizer="adam", epochs=2000, lr=1e-03,
                             dropout_rate=0.1, lr_decay=0.95, lr_patience=10, es_patience=100)



summary.NAMLS(nam_housing,
               data = ca_housing,             # DataFrame mit x-Spalten + Zielspalte
               target_col = "target", # Name der Zielspalte
               pm1_scaler = pm1,        # für [-1, 1] Skalierung
               target_mean = norm_targets$mean,  # für Ziel-Inverse-Transform
               target_sd = norm_targets$sd,      # für Ziel-Inverse-Transform
               show_plot = TRUE,
               yscale = "auto",       # "auto" | "log" | "robust"
               cap_quantile = 0.99,
               drop_first = 1,
               feature_plots = TRUE,  # partielle Effektplots bei >1 Features
               max_features = n_inputs,
               ci_z = 1.96)

dnn_housing <- train.DNN(ca_housing_loader, targets_std_full, val_ca_housing, c(32, 64, 128, 256),
                         optimizer="adam", epochs=2000, lr=1e-03, es_patience=100)



# saveRDS(nam_housing, file = "nam_housing.rds")



# Insurance Data ----------------------------------------------------------
# The Insurance Data is provided within the library as well

data(insurance)

insurance_enc <- one_hot_encode(insurance,  drop_first=TRUE)

dummy_cols  <- detect_dummy_cols(insurance_enc)
dummy_cols

input_vars <- insurance_enc[, setdiff(names(insurance_enc), "charges"), drop = FALSE]

insurance_split <- random_split(input_vars, normalization=FALSE)

targets_insurance <- insurance$charges

train_insurance <- insurance_split$train


# ---- Prepocessing ---- #
# features will be scaled to [-1, 1] excluding dummy columns
pm1 <- pm1_scaler(train_insurance)
pm1

train_insurance <- dummy_pm1_wrapper(train_insurance, pm1, dummy_cols)
# sanity check
sapply(train_insurance, min)
sapply(train_insurance, max)

# Ziehen von val/test splits
val_insurance <- insurance_split$validation
val_targets_insurance <- targets_insurance[as.integer(rownames(val_insurance))]
# [-1, 1] Scaling
val_insurance <- dummy_pm1_wrapper(val_insurance, pm1, dummy_cols)
sapply(val_insurance, min)
sapply(val_insurance, max)

test_insurance <- insurance_split$test
test_targets_insurance <- targets_insurance[as.integer(rownames(test_insurance))]
# [-1, 1] Scaling
test_insurance <- dummy_pm1_wrapper(test_insurance, pm1, dummy_cols)
sapply(test_insurance, min)
sapply(test_insurance, max)

# targets will be standard normalized
train_targets <- targets_insurance[as.integer(rownames(train_insurance))]

norm_targets <- normalize_targets(train_targets, val_targets_insurance, test_targets_insurance)
# reassign normalized targets
train_targets_insurance <- norm_targets$train
val_targets_insurance <- norm_targets$validation
test_targets_insurance <- norm_targets$test

# put them back into one vector by mapping them to their original indices
targets_std_full <- rep(NA_real_, length(targets_insurance))
targets_std_full[as.integer(rownames(train_insurance))] <- train_targets_insurance
targets_std_full[as.integer(rownames(val_insurance))]   <- val_targets_insurance
targets_std_full[as.integer(rownames(test_insurance))]  <- test_targets_insurance

n_inputs <- length(input_vars)

insurance_loader <- DataLoader(train_insurance, batch_size = 256)

nam_insurance <- train_namls(insurance_loader, targets_std_full, length(input_vars),  c(64), t(val_insurance), val_targets_insurance,
                           optimizer="adam", epochs=2000, lr=1e-03,
                           dropout_rate=0.5, lr_decay=0.5, lr_patience=10, es_patience=100)

summary.NAMLS(nam_insurance,
              data = insurance_enc,             # DataFrame mit x-Spalten + Zielspalte
              target_col = "charges", # Name der Zielspalte
              dummy_cols = dummy_cols,
              pm1_scaler = pm1,        # für [-1, 1] Skalierung
              target_mean = norm_targets$mean,  # für Ziel-Inverse-Transform
              target_sd = norm_targets$sd,      # für Ziel-Inverse-Transform
              show_plot = TRUE,
              yscale = "auto",       # "auto" | "log" | "robust"
              cap_quantile = 0.99,
              drop_first = 1,
              feature_plots = TRUE,  # partielle Effektplots bei >1 Features
              max_features = n_inputs,
              ci_z = 1.96)

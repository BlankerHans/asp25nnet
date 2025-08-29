#' This script follows the model applications of Thielman et al. (2020)


# California Housing Data -------------------------------------------------
# The California Housing Data is provided within the library

data(ca_housing)

reduced_df <- ca_housing[, c("MedInc", "HouseAge", "target"), drop = FALSE]
input_vars <- reduced_df[, setdiff(names(reduced_df), "target"), drop = FALSE]

ca_housing_split <- random_split(input_vars, normalization=FALSE)

targets_ca_housing <- ca_housing$target

train_ca_housing <- ca_housing_split$train

# ---- Prepocessing ---- #
# targets will be standard normalized
train_targets <- targets_ca_housing[as.integer(rownames(train_ca_housing))]

# features will be scaled to [-1, 1]
pm1 <- pm1_scaler(train_ca_housing)
pm1

train_ca_housing <- transform_pm1(train_ca_housing, pm1)
View(train_ca_housing)

val_ca_housing <- ca_housing_split$validation
# val_targets vielleicht noch automatisch erkennen mit in train aufnehmen!?
val_targets_ca_housing <- targets_ca_housing[as.integer(rownames(val_ca_housing))]

ca_housing_loader <- DataLoader(train_ca_housing, batch_size = 256)

nam_housing <- train_namls(ca_housing_loader, targets_ca_housing, 1,  c(50), t(val_ca_housing), val_targets_ca_housing,
                           optimizer="adam", epochs=2000, lr=0.001,
                           dropout_rate=0.1, lr_decay=0.95, lr_patience=10)
summary.NAMLSS(nam_housing,
               data = reduced_df,             # DataFrame mit x-Spalten + Zielspalte
               target_col = "target",      # Name der Zielspalte
               show_plot = TRUE,
               yscale = "robust",       # "auto" | "log" | "robust"
               cap_quantile = 0.99,
               drop_first = 1,
               feature_plots = FALSE,  # partielle Effektplots bei >1 Features
               max_features = 4,
               ci_z = 1.96)




# Insurance Data ----------------------------------------------------------
# The Insurance Data is provided within the library as well

data(insurance)

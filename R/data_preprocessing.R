#' Data Preprocessing Functions
#' This script contains functions for data preprocessing steps described by
#' Thielmann et al. (2020), including scaling to [-1, 1] and standard normalizing
#' of target variables.

# --- Scaling [-1, 1] ----------------------------------------------------

# pro Feature a_j, b_j aus train
pm1_scaler <- function(X_train, eps = 1e-8) {
  X_train <- as.matrix(X_train)
  a <- apply(X_train, 2, min, na.rm = TRUE)
  b <- apply(X_train, 2, max, na.rm = TRUE)
  list(a = a, b = b, eps = eps)
}

# Transformation mit zuvor gefitteten Parametern
transform_pm1 <- function(X, scaler, clip = TRUE) {
  #X <- as.matrix(X)
  # Spaltenreihenfolge anpassen / sicherstellen
  # missing_cols <- setdiff(names(scaler$a), colnames(X))
  # if (length(missing_cols) > 0) {
  #   # Falls Spalten fehlen, füge konstante Spalten mit dem Train-Min ein (-> wird zu -1)
  #   for (m in missing_cols) X[, m] <- scaler$a[[m]]
  #   X <- X[, names(scaler$a), drop = FALSE]
  # } else {
  #   X <- X[, names(scaler$a), drop = FALSE]
  # }
  a <- scaler$a
  b <- scaler$b

  den <- pmax(b - a, scaler$eps)
  #Z <- sweep(sweep(X, 2, a, "-"), 2, den, "/")
  Z <- 2 * ((X-a)/(b-a)) - 1

  # Konstante Features (b==a) sauber auf 0 setzen
  const_cols <- which((b - a) < scaler$eps)
  if (length(const_cols) > 0) Z[, const_cols] <- 0

  if (clip) Z <- pmin(1, pmax(-1, Z))
  Z
}

# --- Target-Standardisierung --------------------------------------------

normalize_targets <- function(train_targets, val_targets = NULL, test_targets = NULL) {
  train_mean <- mean(train_targets, na.rm = TRUE)
  train_sd   <- sd(train_targets,   na.rm = TRUE)
  if (isTRUE(is.na(train_sd)) || train_sd == 0) stop("Target SD is zero or NA on training split.")

  res <- list(
    train = (train_targets - train_mean) / train_sd,
    mean  = train_mean,
    sd    = train_sd
  )
  if (!is.null(val_targets))  res$validation <- (val_targets  - train_mean) / train_sd
  if (!is.null(test_targets)) res$test       <- (test_targets - train_mean) / train_sd
  res
}

inverse_target <- function(t_std, mean, sd) mean + sd * t_std




# to do one-hot encoder

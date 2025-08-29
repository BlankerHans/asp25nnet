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


transform_pm1 <- function(X, scaler, clip = FALSE) {
  rn <- rownames(X)
  # Spalten in Train-Reihenfolge
  X <- as.data.frame(X)[, names(scaler$a), drop = FALSE]

  a <- scaler$a; b <- scaler$b
  den <- b - a

  Z <- 2 * sweep(sweep(as.matrix(X), 2, a, "-"), 2, ifelse(den == 0, 1, den), "/") - 1
  # konstante Spalten exakt 0
  if (any(den == 0)) Z[, den == 0] <- 0
  if (clip) {
    Z[Z >  1] <-  1
    Z[Z < -1] <- -1
  } # nur relevant für val und test split

  dimnames(Z) <- list(rn, names(a))
  Z <- as.data.frame(Z)

  #attr(out, "normalization_params") <- attr(X, "normalization_params", exact = TRUE)

  return(Z)
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

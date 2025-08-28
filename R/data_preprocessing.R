#' Data Preprocessing Functions
#' This script contains functions for data preprocessing steps described by
#' Thielmann et al. (2020), including scaling to [-1, 1] and standard normalizin
#' g of target variables.

scale_minus_one_to_one <- function(x) {
  x_min <- min(x, na.rm = TRUE)
  x_max <- max(x, na.rm = TRUE)
  2 * (x - x_min) / (x_max - x_min) - 1
}


normalize_targets <- function(train_targets, val_targets = NULL, test_targets = NULL) {

  train_mean <- mean(train_targets, na.rm = TRUE)
  train_sd <- sd(train_targets, na.rm = TRUE)

  train_norm <- (train_targets - train_mean) / train_sd

  result <- list(
    train = train_norm,
    mean = train_mean,
    sd = train_sd
  )

  if (!is.null(val_targets)) {
    result$validation <- (val_targets - train_mean) / train_sd
  }

  if (!is.null(test_targets)) {
    result$test <- (test_targets - train_mean) / train_sd
  }

  return(result)
}

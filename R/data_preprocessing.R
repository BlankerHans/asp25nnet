#' Data preprocessing utilities (Thielmann et al., 2020)
#'
#' A compact set of helper functions for common preprocessing steps:
#' scaling features to the interval \code{[-1, 1]}, standardizing target
#' variables using training statistics, one-hot encoding of categorical
#' features, and preserving dummy (0/1) columns during scaling.
#'
#' @details
#' **Scaling to \code{[-1, 1]}**
#'
#' \itemize{
#'   \item \code{pm1_scaler(X_train, eps)} estimates per-feature minima
#'     \eqn{a_j} and maxima \eqn{b_j} on the training split.
#'   \item \code{transform_pm1(X, scaler, clip)} applies the linear mapping
#'     \eqn{z_j = 2\,\frac{x_j - a_j}{\max(b_j - a_j, 1)} - 1}. Constant
#'     columns (\eqn{a_j = b_j}) are set exactly to \eqn{0}. Columns are
#'     aligned to the training order; optional clipping confines outputs to
#'     \code{[-1, 1]} for validation/test data.
#' }
#'
#' **Target standardization**
#'
#' \itemize{
#'   \item \code{normalize_targets(train, val, test)} returns standardized
#'     targets using mean and standard deviation estimated on \emph{training}
#'     only; aborts if the training standard deviation is \eqn{0} or \code{NA}.
#'   \item \code{inverse_target(t\_std, mean, sd)} back-transforms via
#'     \eqn{\text{mean} + \text{sd} \times t_{\text{std}}}.
#' }
#'
#' **Categorical encoding**
#'
#' \itemize{
#'   \item \code{one_hot_encode(data, cat_cols, ordered_levels, drop_first)}
#'     creates dummy variables per categorical column using
#'     \code{model.matrix(~ col - 1)}; numeric columns are kept unchanged.
#'     Optionally drops the first level per column to avoid redundancy.
#' }
#'
#' **Dummy handling with scaling**
#'
#' \itemize{
#'   \item \code{detect_dummy_cols(df)} detects columns that contain only
#'     \code{0}/\code{1} (allowing \code{NA}) and thus represent existing
#'     binary indicators.
#'   \item \code{dummy_pm1_wrapper(X, scaler, dummy_cols, clip)} scales
#'     continuous features via \code{transform_pm1()} and then restores the
#'     specified dummy columns from the original \code{X} so that binary
#'     indicators remain intact.
#' }
#'
#' @section Notes:
#' \itemize{
#'   \item All parameters for scaling/standardization are estimated on the
#'     training split and reused for validation/test to prevent leakage.
#'   \item \code{transform_pm1()} aligns columns to the training order; missing
#'     training columns are created as \code{NA} before transformation; unknown
#'     columns are ignored.
#'   \item \code{one_hot_encode()} converts character columns to \code{factor}
#'     automatically; supply \code{ordered_levels} for reproducible level order.
#' }
#'
#' @references
#' Thielmann, A., et al. (2020).
#'
#' @aliases
#' pm1_scaler
#' transform_pm1
#' normalize_targets
#' inverse_target
#' one_hot_encode
#' detect_dummy_cols
#' dummy_pm1_wrapper
#'
#' @keywords preprocessing normalization scaling encoding
#' @name data_preprocessing
#' @md
NULL

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

one_hot_encode <- function(data, cat_cols = NULL, ordered_levels = NULL, drop_first = FALSE) {
  data <- as.data.frame(data)

  # Auto-detect categorical columns
  if (is.null(cat_cols)) {
    cat_cols <- names(data)[sapply(data, function(x) is.factor(x) || is.character(x))]
  }

  if (length(cat_cols) == 0) {
    return(data)
  }

  # Separate numerical and categorical
  num_cols <- setdiff(names(data), cat_cols)
  data_num <- data[, num_cols, drop = FALSE]

  # One-hot encode each categorical column
  encoded_list <- list()

  for (col in cat_cols) {
    # Apply custom ordering if specified
    if (!is.null(ordered_levels[[col]])) {
      data[[col]] <- factor(data[[col]], levels = ordered_levels[[col]], ordered = TRUE)
    } else if (is.character(data[[col]])) {
      data[[col]] <- as.factor(data[[col]])
    }

    # Check for missing levels
    if (any(is.na(data[[col]]) & !is.na(data[[col]]))) {
      warning(sprintf("Unknown categories in column '%s' will be set to NA", col))
    }

    # Create dummy variables
    if (drop_first && nlevels(data[[col]]) > 1) {
      dummies <- model.matrix(~ data[[col]] - 1)[, -1, drop = FALSE]
      colnames(dummies) <- paste0(col, "_", levels(data[[col]])[-1])
    } else {
      dummies <- model.matrix(~ data[[col]] - 1)
      colnames(dummies) <- paste0(col, "_", levels(data[[col]]))
    }

    encoded_list[[col]] <- as.data.frame(dummies)
  }

  # Combine all
  if (length(encoded_list) > 0) {
    data_encoded <- do.call(cbind, encoded_list)
    result <- cbind(data_num, data_encoded)
  } else {
    result <- data_num
  }

  return(result)
}


# 1) Dummy-Spalten erkennen (0/1 numerisch)
detect_dummy_cols <- function(df) {
  names(df)[sapply(df, function(x)
    is.numeric(x) && all(x %in% c(0, 1) | is.na(x)))]
}

# 2) Wrapper: skaliert wie bisher, überschreibt danach Dummies mit Original
dummy_pm1_wrapper <- function(X, scaler, dummy_cols, clip = FALSE) {
  X <- as.data.frame(X)
  Z <- transform_pm1(X, scaler, clip = clip)
  keep <- intersect(dummy_cols, names(Z))
  if (length(keep)) Z[keep] <- X[keep]
  Z
}


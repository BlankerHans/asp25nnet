#' Split dataset into training, validation and test sets with optional normalization
#'
#' This function splits a given dataset into training, validation, and test subsets
#' according to the specified proportions. It optionally normalizes all subsets based
#' on the mean and standard deviation of the training set.
#'
#' @param data A data frame or matrix containing the dataset to split. Just use the
#' features, not the target variable.
#' @param split Numeric vector of length 3 specifying the proportions for the training,
#' validation, and test sets. Must sum to 1. Default is c(0.6, 0.2, 0.2).
#' @param normalization Logical indicating whether to normalize the subsets based on
#'  mean and standard deviation of the training data. Default is TRUE.
#'
#' @return A list with the following components:
#' \describe{
#'   \item{train}{Training subset (normalized if \code{normalization = TRUE}).}
#'   \item{validation}{Validation subset (normalized if \code{normalization = TRUE}).}
#'   \item{test}{Test subset (normalized if \code{normalization = TRUE}).}
#'   \item{normalization_params}{List with \code{mean} and \code{sd} vectors of training data used for normalization. Only returned if \code{normalization = TRUE}.}
#' }
#'
#' @examples
#' \dontrun{
#'   data(iris)
#'   result <- train_val_test(iris[, 1:4], split = c(0.7, 0.15, 0.15), normalization = TRUE)
#'   str(result)
#' }
#'
#' @export

random_split <- function(data, split = c(0.6, 0.2, 0.2), normalization = TRUE) {

  # Check type and length of split
  if (!is.numeric(split) || length(split) != 3) {
    stop("`split` must be a numeric vector of length 3, e.g. c(0.6, 0.2, 0.2).")
  }

  # Check range
  if (any(split < 0) || any(split > 1)) {
    stop("All entries in `split` must be between 0 and 1.")
  }

  # Check sum (must be exactly 1)
  if (sum(split) != 1) {
    stop("The sum of `split` must be exactly 1 (you have ", sum(split), ").")
  }

  # Shuffle indices
  n <- nrow(data)
  idx_all <- sample(n)

  # Compute sizes
  n_train <- floor(split[1] * n)
  n_val   <- floor(split[2] * n)
  n_test  <- n - n_train - n_val  # ensures all rows are assigned

  # Split indices
  train_idx <- idx_all[1:n_train]
  val_idx   <- idx_all[(n_train + 1):(n_train + n_val)]
  test_idx  <- idx_all[(n_train + n_val + 1):n]

  train <- data[train_idx, , drop = FALSE]
  validation <- data[val_idx, , drop = FALSE]
  test <- data[test_idx, , drop = FALSE]

  # Normalization if requested
  if (!is.logical(normalization) || length(normalization) != 1) {
    stop("`normalization` must be TRUE or FALSE.")
  }
    if (normalization) {
      mean_train <- apply(train, 2, mean) # calculates means of all train columns
      sd_train <- apply(train, 2, sd) # calculates sd of all train columns

      #function for scaling data with specified mean and sd
      normalize <- function(x, mean_vec, sd_vec) {
        scale(x, center =mean_vec, scale = sd_vec)
      }

      #normalize train, val & test
      train_norm <- normalize(train, mean_train, sd_train)
      val_norm <- normalize(validation, mean_train, sd_train)
      test_norm <- normalize(test, mean_train, sd_train)

      return(list(
        train = train_norm,
        validation = val_norm,
        test = test_norm,
        normalization_params = list(mean = mean_train, sd = sd_train)
      ))

    } else {
      return(list(
        train = train,
        validation = validation,
        test = test
      ))
    }
}

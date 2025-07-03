#' Split a data frame into training, validation, and test sets
#'
#' Randomly shuffles the rows of a data frame and splits them into
#' training, validation, and test subsets according to specified proportions.
#'
#' @param data A data frame or matrix to split.
#' @param split Numeric vector of length 3 specifying proportions for train,
#'   validation, and test sets. Must sum to exactly 1.
#' @param normalization Logical, whether to z-score normalize the data.
#'
#' @return A list with three elements: \code{train}, \code{validation},
#'   and \code{test}, each containing the corresponding subset of rows.
#'
#' @examples
#' set.seed(42)
#' df <- data.frame(x = rnorm(100), y = rnorm(100))
#' res <- train_val_test(df, split = c(0.6, 0.2, 0.2))
#' str(res)
train_val_test <- function(data, split = c(0.6, 0.2, 0.2), normalization = TRUE) {

  # 1) Check type and length of split
  if (!is.numeric(split) || length(split) != 3) {
    stop("`split` must be a numeric vector of length 3, e.g. c(0.6, 0.2, 0.2).")
  }

  # 2) Check range
  if (any(split < 0) || any(split > 1)) {
    stop("All entries in `split` must be between 0 and 1.")
  }

  # 3) Check sum (must be exactly 1)
  if (sum(split) != 1) {
    stop("The sum of `split` must be exactly 1 (you have ", sum(split), ").")
  }

  n <- nrow(data)

  # 4) Normalization if requested
  if (!is.logical(normalization) || length(normalization) != 1) {
    stop("`normalization` must be TRUE or FALSE.")
  }
  if (normalization) {
    rn <- rownames(data)
    data <- scale(data)
    rownames(data) <- rn
  }

  # 5) Shuffle indices
  idx_all <- sample(n)

  # 6) Compute sizes
  n_train <- floor(split[1] * n)
  n_val   <- floor(split[2] * n)
  n_test  <- n - n_train - n_val  # ensures all rows are assigned

  # 7) Split indices
  train_idx <- idx_all[1:n_train]
  val_idx   <- idx_all[(n_train + 1):(n_train + n_val)]
  test_idx  <- idx_all[(n_train + n_val + 1):n]

  # 8) Return subsets
  return(list(
    train = data[train_idx, , drop = FALSE],
    validation = data[val_idx, , drop = FALSE],
    test = data[test_idx, , drop = FALSE]
  ))
}

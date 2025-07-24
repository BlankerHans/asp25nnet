#' Random Split of a Dataset
#'
#' Splits a dataset into training and test sets based on the specified proportions.
#' Optionally applies normalization (mean centering and scaling) to all columns.
#'
#' @param data A numeric matrix or data frame to split.
#' @param split Numeric vector of length 2 specifying the proportions for the train and test sets. Default is \code{c(0.8, 0.2)}.
#' @param normalization Logical value indicating whether to normalize the data. Default is \code{TRUE}.
#'
#' @return A list with two elements:
#' \item{train}{The training set as a matrix.}
#' \item{test}{The test set as a matrix.}
#'
#' @details
#' If the sum of \code{split} is less than 1, the remaining proportion of rows will be excluded from both splits.
#' Normalization applies scaling across all rows before splitting.
#' Rows are selected sequentially without shuffling.
#'
#' @examples
#' data <- matrix(rnorm(100), ncol = 5)
#' result <- random_split(data, split = c(0.7, 0.3), normalization = TRUE)
#' dim(result$train)
#' dim(result$test)
#'
#' @export
random_split <- function(data, split=c(0.8, 0.2), normalization=TRUE) {

  # prüft ob Split-Vektor zulässig ist und setzt 0.8,0.2 als default

  # 1) Typ / Länge prüfen
  if (!is.numeric(split) || length(split) != 2) {
    stop("`split` muss ein numerischer Vektor der Länge 2 sein, z.B. c(0.8, 0.2).")
  }
  # 2) Wertebereich prüfen
  if (any(split < 0) || any(split > 1)) {
    stop("Alle Einträge in `split` müssen zwischen 0 und 1 liegen.")
  }
  # 3) Summe prüfen
  if (sum(split) > 1) {
    stop("Die Summe von `split` darf maximal 1.0 sein (du hast ", sum(split), ").")
  }

  # Bestimmt Größe von Trainings- und Testdatensatz
  n <- nrow(data)
  n_train <- floor(split[1] * n)
  n_test  <- floor(split[2] * n) # oder besser 1-n_train

  # normalization
  if (!is.logical(normalization) || length(normalization) != 1) {
    stop("`normalization` must be TRUE or FALSE")
  }
  if (normalization) {
    rn <- rownames(data)
    data <- scale(data)
    rownames(data) <- rn
  }

  # ohne shuffle
  train <- data[1:n_train, , drop = FALSE]
  test <- data[(n_train + 1):n, , drop = FALSE]


  return(list(
    train = train,
    test  = test
  ))
}

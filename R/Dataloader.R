#' DataLoader: Load data in batches with optional shuffling
#'
#' Splits a dataset into batches of size \code{batch_size}.
#' Optionally shuffles the data before batching.
#' Each batch is transposed and returned along with the indices of included data points.
#'
#' @param data A data frame or matrix to be processed in batches.
#' @param batch_size Integer specifying the size of each batch (default is 32).
#' @param shuffle Logical, whether to shuffle the data before batching (default is TRUE).
#'
#' @return A list of lists, each containing:
#' \itemize{
#'   \item \code{batch}: A matrix (features × batch size) with the batch data.
#'   \item \code{idx}: An integer vector with the original indices of data points in the batch.
#' }
#'
#' @details
#' The function generates start indices for the batches (e.g., 1, 33, 65, ... with \code{batch_size = 32}) and extracts the corresponding rows from \code{data}.
#' Each batch is transposed (columns represent individual data points) and returned along with their indices.
#'
#' @examples
#' data <- matrix(1:1000, ncol = 10)
#' batches <- DataLoader(data, batch_size = 50, shuffle = FALSE)
#' str(batches[[1]])
#'
#' @export
DataLoader <- function(data, batch_size=32, shuffle=TRUE) {

  # Shuffle data, generate start indices for batches (e.g. 1, 33, 65 with batch_size=32)
  # Extract rows for each batch, transpose batches and return them with data point indices

  data <- as.matrix(data)

  if (shuffle) {
    data <- data[sample(nrow(data)), , drop = FALSE]
  }

  n <- nrow(data)
  # Start indices for batches: 1, 1+batch_size, 1+2*batch_size, …
  starts <- seq(1, n, by = batch_size)

  # Generate a list of batches
  batches <- lapply(starts, function(i) {
    mat <- data[i:min(i+batch_size-1, n), , drop = FALSE]
    mat_t <- t(mat)
    idx   <- as.integer(colnames(mat_t))
    list(
      batch = mat_t,
      idx = idx
    )
  })

  return(batches)
}

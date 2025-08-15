#' Get Layer Dimensions for a Neural Network
#'
#' Determines the input, hidden, and output layer dimensions based on the input matrix
#' and specified configuration.
#'
#' @param X A numeric matrix or vector representing input data. If a matrix, rows correspond to features and columns to observations.
#' @param out_dim Integer specifying the output dimension (e.g., number of output units).
#' @param hidden_neurons Integer specifying the number of neurons in the hidden layer.
#'
#' @return A named list with elements:
#' \item{n_x}{Number of input features (rows of \code{X}).}
#' \item{n_h}{Number of hidden neurons.}
#' \item{n_y}{Number of output neurons.}
#'
#' @examples
#' X <- matrix(rnorm(20), nrow = 5)
#' dims <- getLayerDimensions(X, out_dim = 2, hidden_neurons = 4)
#' dims$n_x
#'
#' @export
getLayerDimensions <- function(X, out_dim, hidden_neurons) {
  n_x <- dim(X)[1]
  n_h <- hidden_neurons
  n_y <- out_dim

  dimensions_list <- list("n_x" = n_x,
                          "n_h" = n_h,
                          "n_y" = n_y)

  return(dimensions_list)
}

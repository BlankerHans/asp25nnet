#' Get layer dimensions for variable-size network
#'
#' @param X Input data matrix
#' @param out_dim Output dimension
#' @param hidden_neurons Vector of neurons per hidden layer, e.g., c(50, 30, 20) chronologically ordered flowing from input to output, i.e. left to right
#' @return List with network dimensions
#' @export
getLayerDimensions_variable <- function(X, out_dim, hidden_neurons) {
  n_x <- dim(X)[1]

  # Ensure hidden_neurons is a vector
  if (!is.vector(hidden_neurons) || length(hidden_neurons) == 0) {
    stop("hidden_neurons must be a non-empty vector, e.g., c(50, 30, 20) chronologically ordered flowing from input to output, i.e. left to right!")
  }

  list(
    n_x = n_x,
    n_h = hidden_neurons,
    n_y = out_dim,
    n_layers = length(hidden_neurons)
  )
}

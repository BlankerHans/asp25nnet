#' Backpropagation for a Single-Hidden-Layer Neural Network
#'
#' Computes the gradients of the parameters for a neural network with one hidden layer,
#' assuming a Gaussian likelihood with mean and log standard deviation outputs.
#'
#' @param X A numeric matrix of shape (p × m), where p is the number of input features and m is the batch size.
#' @param y Numeric vector of length m containing the target values.
#' @param cache A list containing intermediate values from the forward pass:
#'   \code{Z1}, \code{A1}, \code{Z2}, \code{mu}, \code{log_sigma}.
#' @param params A list of model parameters:
#'   \code{W1}, \code{b1}, \code{W2}, \code{b2}.
#'
#' @return A list of gradients with respect to the parameters:
#' \itemize{
#'   \item \code{dW1}: Gradient matrix for the first layer weights.
#'   \item \code{db1}: Gradient vector for the first layer biases.
#'   \item \code{dW2}: Gradient matrix for the second layer weights.
#'   \item \code{db2}: Gradient vector for the second layer biases.
#' }
#'
#' @details
#' This function performs backpropagation using derivatives of the negative log-likelihood
#' under a normal distribution. The output layer returns both the predicted mean (\code{mu})
#' and log standard deviation (\code{log_sigma}). Gradients are averaged over the batch.
#'
#' @examples
#' # Example usage:
#' X <- matrix(rnorm(20), nrow = 4)
#' y <- rnorm(5)
#' params <- list(
#'   W1 = matrix(rnorm(8), nrow = 2),
#'   b1 = matrix(0, nrow = 2),
#'   W2 = matrix(rnorm(4), nrow = 2),
#'   b2 = matrix(0, nrow = 2)
#' )
#' cache <- forward_onehidden(X, params)
#' grads <- backprop_onehidden(X, y, cache, params)
#'
#' @export
backprop_onehidden <- function(X, y, cache, params) {

  m <- ncol(X)

  mu        <- as.numeric(cache$mu)
  log_sigma <- as.numeric(cache$log_sigma)
  sigma2    <- exp(2 * log_sigma)

  delta_mu  <- (mu - y) / sigma2
  delta_eta <- 1 - (y - mu)^2 / sigma2

  delta2 <- rbind(delta_mu, delta_eta)

  dW2 <- (delta2 %*% t(cache$A1)) / m
  db2 <- rowSums(delta2) / m

  dA1 <- t(params$W2) %*% delta2
  dZ1 <- dA1 * (cache$Z1 > 0)

  dW1 <- (dZ1 %*% t(X)) / m
  db1 <- rowSums(dZ1) / m

  list(
    dW1 = dW1,
    db1 = matrix(db1, ncol = 1),
    dW2 = dW2,
    db2 = matrix(db2, ncol = 1)
  )
}

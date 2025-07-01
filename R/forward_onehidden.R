#' Forward Pass Through a Single Hidden Layer Neural Network
#'
#' Computes the forward propagation of inputs through a neural network with one hidden ReLU layer,
#' returning intermediate activations and outputs.
#'
#' @param X A numeric matrix of input data, where columns are observations and rows are features.
#' @param params A named list of network parameters containing:
#'   \itemize{
#'     \item{\code{W1}: Weight matrix for the hidden layer.}
#'     \item{\code{b1}: Bias vector for the hidden layer.}
#'     \item{\code{W2}: Weight matrix for the output layer.}
#'     \item{\code{b2}: Bias vector for the output layer.}
#'   }
#'
#' @return A named list containing:
#'   \item{Z1}{Pre-activation matrix of the hidden layer (before ReLU).}
#'   \item{A1}{Activation matrix of the hidden layer (after ReLU).}
#'   \item{Z2}{Output pre-activation matrix.}
#'   \item{mu}{Estimated mean vector (first row of Z2).}
#'   \item{log_sigma}{Estimated log standard deviation vector (second row of Z2).}
#'
#' @details
#' The function computes:
#' \enumerate{
#'   \item Hidden pre-activation: \eqn{Z1 = W1 \%*\% X + b1}
#'   \item Hidden activation: \eqn{A1 = ReLU(Z1)}
#'   \item Output pre-activation: \eqn{Z2 = W2 \%*\% A1 + b2}
#' }
#'
#' Biases are expanded to match the batch size by matrix multiplication.
#'
#' @examples
#' X <- matrix(rnorm(10), nrow = 5)
#' params <- list(
#'   W1 = matrix(rnorm(8), nrow = 4),
#'   b1 = matrix(0, nrow = 4, ncol = 1),
#'   W2 = matrix(rnorm(8), nrow = 2),
#'   b2 = matrix(0, nrow = 2, ncol = 1)
#' )
#' cache <- forward_onehidden(X, params)
#' cache$mu
#'
#' @export
forward_onehidden <- function(X, params) {
  ones <- matrix(1, nrow = 1, ncol = dim(X)[2])
  Z1 <- params$W1 %*% X + params$b1 %*% ones
  A1 <- ReLU(Z1)
  Z2 <- params$W2 %*% A1 + params$b2 %*% ones
  mu_hat  <- Z2[1, , drop = FALSE]
  log_sigma_hat <- Z2[2, , drop = FALSE]

  cache <- list(
    "Z1" = Z1,
    "A1" = A1,
    "Z2" = Z2,
    "mu" = mu_hat,
    "log_sigma" = log_sigma_hat
  )

  return(cache)
}

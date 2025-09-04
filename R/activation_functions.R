#' Sigmoid Activation Function
#'
#' Computes the sigmoid activation \eqn{1 / (1 + exp(-x))}.
#'
#' @param x A numeric vector, matrix, or array.
#' @return A numeric object of the same shape as \code{x} with the sigmoid applied element-wise.
#' @export
#'
#' @examples
#' sigmoid(0)
#' sigmoid(c(-2, 0, 2))
sigmoid <- function(x) {
  return(1 / (1 + exp(-x)))
}

#' ReLU Activation Function
#'
#' Computes the Rectified Linear Unit (ReLU), defined as \eqn{\max(0, x)}.
#'
#' @param x A numeric vector or matrix.
#' @return A numeric object with the same shape and column names as \code{x}, with ReLU applied element-wise.
#' @export
#'
#' @examples
#' ReLU(c(-1, 0, 1, 5))
#' ReLU(matrix(c(-2, 0, 2), nrow = 1, dimnames = list(NULL, c("a", "b", "c"))))
ReLU <- function(x) {
  out <- pmax(0, x)
  cn <- colnames(x)
  dim(out) <- dim(x)
  colnames(out) <- cn
  return(out)
}

#' Softplus Activation Function
#'
#' Computes the softplus activation \eqn{log(1 + exp(x))}.
#'
#' @param x A numeric vector, matrix, or array.
#' @return A numeric object of the same shape as \code{x} with softplus applied element-wise.
#' @export
#'
#' @examples
#' Softplus(0)
#' Softplus(c(-2, 0, 2))
Softplus <- function(x) {
  pmax(x, 0) + log1p(exp(-abs(x)))
}


inv_softplus <- function(y) log(expm1(y)) # numerische stabilität in NAMLSS

#' ELU Activation Function
#'
#' Computes the Exponential Linear Unit (ELU): \eqn{x} if \eqn{x > 0}, else \eqn{alpha * (exp(x) - 1)}.
#'
#' @param x A numeric vector, matrix, or array.
#' @param alpha A numeric scalar, the ELU scale for negative values (default is 1).
#' @return A numeric object of the same shape as \code{x} with ELU applied element-wise.
#' @export
#'
#' @examples
#' ELU(c(-1, 0, 1))
#' ELU(-2:2, alpha = 1.5)
ELU <- function(x, alpha = 1) {
  return(ifelse(x > 0, x, alpha * (exp(x) - 1)))
}


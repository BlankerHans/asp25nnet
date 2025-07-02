#' Initialize Neural Network Parameters
#'
#' Creates randomly initialized weight matrices and zero bias vectors for a single-hidden-layer neural network.
#'
#' @param dimensions_list A list specifying the network dimensions. Must contain:
#'   \code{n_x}: Number of input features,
#'   \code{n_h}: Number of hidden neurons,
#'   \code{n_y}: Number of output units.
#' @param seed Integer random seed for reproducibility. Default is 42.
#'
#' @return A list containing:
#' \itemize{
#'   \item \code{W1}: Weight matrix of shape (n_h × n_x).
#'   \item \code{b1}: Bias vector of shape (n_h × 1).
#'   \item \code{W2}: Weight matrix of shape (n_y × n_h).
#'   \item \code{b2}: Bias vector of shape (n_y × 1).
#' }
#'
#' @details
#' Weights are initialized with random values drawn from a normal distribution
#' with mean 0 and standard deviation 0.1. Bias vectors are initialized to zeros.
#'
#' @examples
#' dims <- list(n_x = 5, n_h = 10, n_y = 2)
#' params <- init_params(dims)
#'
#' @export
init_params <- function(dimensions_list, seed=42) {
  set.seed(seed)
  list(
    W1 = matrix(rnorm(dimensions_list$n_h * dimensions_list$n_x, sd = 0.1),
                nrow = dimensions_list$n_h,
                ncol = dimensions_list$n_x),
    b1 = matrix(0, nrow = dimensions_list$n_h, ncol = 1),
    W2 = matrix(rnorm(dimensions_list$n_y * dimensions_list$n_h, sd = 0.1),
                nrow = dimensions_list$n_y,
                ncol = dimensions_list$n_h),
    b2 = matrix(0, nrow = dimensions_list$n_y, ncol = 1)
  )
}

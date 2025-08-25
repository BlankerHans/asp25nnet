#' Initialize parameters for multi-layer network with variable layer sizes
#'
#' @param dimensions_list List with n_x (input), n_h (vector of hidden neurons per layer), n_y (output)
#' @param seed Random seed for reproducibility
#' @return List of weight matrices and bias vectors for all layers
#' @export
init_params <- function(dimensions_list, seed = 42) {
  if (!is.null(seed)) set.seed(seed)

  n_x <- dimensions_list$n_x  # Input dimension
  n_h <- dimensions_list$n_h  # Vector of hidden layer sizes
  n_y <- dimensions_list$n_y  # Output dimension

  # Ensure n_h is a vector
  if (!is.vector(n_h) || length(n_h) == 0) {
    stop("n_h must be a non-empty vector specifying neurons for each hidden layer")
  }

  n_layers <- length(n_h)  # Number of hidden layers
  params <- list()

  # First layer: input -> first hidden
  params$W1 <- matrix(rnorm(n_h[1] * n_x, sd = sqrt(2/n_x)),
                      nrow = n_h[1], ncol = n_x)
  params$b1 <- matrix(0, nrow = n_h[1], ncol = 1)

  # Hidden layers (if more than 1)
  if (n_layers > 1) {
    for (l in 2:n_layers) {
      W_name <- paste0("W", l)
      b_name <- paste0("b", l)
      n_prev <- n_h[l-1]  # Neurons in previous layer
      n_curr <- n_h[l]    # Neurons in current layer

      params[[W_name]] <- matrix(rnorm(n_curr * n_prev, sd = sqrt(2/n_prev)),
                                 nrow = n_curr, ncol = n_prev)
      params[[b_name]] <- matrix(0, nrow = n_curr, ncol = 1)
    }
  }

  # Output layer: last hidden -> output
  W_out_name <- paste0("W", n_layers + 1)
  b_out_name <- paste0("b", n_layers + 1)
  n_last <- n_h[n_layers]  # Neurons in last hidden layer

  params[[W_out_name]] <- matrix(rnorm(n_y * n_last, sd = sqrt(2/n_last)),
                                 nrow = n_y, ncol = n_last)
  params[[b_out_name]] <- matrix(0, nrow = n_y, ncol = 1)

  # Store architecture info for later use
  attr(params, "architecture") <- list(
    n_x = n_x,
    n_h = n_h,
    n_y = n_y,
    n_layers = n_layers
  )

  return(params)
}

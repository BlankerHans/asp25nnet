#' Forward propagation for network with variable layer sizes
#'
#' @param X Input matrix (features × batch_size)
#' @param params Network parameters from init_params
#' @return List with all intermediate values and outputs
#' @export
forward <- function(X, params) {
  # Extract architecture info
  arch <- attr(params, "architecture")
  if (is.null(arch)) {
    stop("Parameters must have architecture attribute. Use init_params().")
  }

  n_layers <- arch$n_layers
  cache <- list()
  ones <- matrix(1, nrow = 1, ncol = ncol(X))

  # First hidden layer
  cache$Z1 <- params$W1 %*% X + params$b1 %*% ones
  cache$A1 <- ReLU(cache$Z1)

  # Additional hidden layers
  if (n_layers > 1) {
    for (l in 2:n_layers) {
      Z_name <- paste0("Z", l)
      A_name <- paste0("A", l)
      A_prev <- cache[[paste0("A", l-1)]]
      W_name <- paste0("W", l)
      b_name <- paste0("b", l)

      cache[[Z_name]] <- params[[W_name]] %*% A_prev + params[[b_name]] %*% ones
      cache[[A_name]] <- ReLU(cache[[Z_name]])
    }
  }

  # Output layer
  A_last <- cache[[paste0("A", n_layers)]]
  W_out <- params[[paste0("W", n_layers + 1)]]
  b_out <- params[[paste0("b", n_layers + 1)]]

  Z_out <- W_out %*% A_last + b_out %*% ones
  cache[[paste0("Z", n_layers + 1)]] <- Z_out

  # Extract mu and log_sigma
  cache$mu <- Z_out[1, , drop = FALSE]
  cache$log_sigma <- Z_out[2, , drop = FALSE]

  # Store architecture for backprop
  cache$n_layers <- n_layers

  return(cache)
}

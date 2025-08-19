#' Backpropagation for network with variable layer sizes
#'
#' @param X Input matrix
#' @param y Target values
#' @param cache Forward propagation cache
#' @param params Network parameters
#' @return List of gradients
#' @export
backprop_variable <- function(X, y, cache, params) {
  B <- ncol(X)
  n_layers <- cache$n_layers
  grads <- list()

  # Output layer gradients
  mu <- as.numeric(cache$mu)
  log_sigma <- as.numeric(cache$log_sigma)
  sigma2 <- exp(2 * log_sigma)

  delta_mu <- (mu - y) / sigma2
  delta_eta <- 1 - (y - mu)^2 / sigma2

  # Gradient for output layer
  delta <- rbind(delta_mu, delta_eta)
  A_last <- cache[[paste0("A", n_layers)]]

  W_out_name <- paste0("W", n_layers + 1)
  dW_out_name <- paste0("dW", n_layers + 1)
  db_out_name <- paste0("db", n_layers + 1)

  grads[[dW_out_name]] <- (delta %*% t(A_last)) / B
  grads[[db_out_name]] <- matrix(rowSums(delta) / B, ncol = 1)

  # Backpropagate through hidden layers
  dA_next <- t(params[[W_out_name]]) %*% delta

  for (l in n_layers:1) {
    Z_name <- paste0("Z", l)
    W_name <- paste0("W", l)
    dW_name <- paste0("dW", l)
    db_name <- paste0("db", l)

    dZ <- dA_next * (cache[[Z_name]] > 0)  # ReLU derivative

    if (l > 1) {
      A_prev <- cache[[paste0("A", l-1)]]
      grads[[dW_name]] <- (dZ %*% t(A_prev)) / B
    } else {
      grads[[dW_name]] <- (dZ %*% t(X)) / B
    }

    grads[[db_name]] <- matrix(rowSums(dZ) / B, ncol = 1)

    if (l > 1) {
      dA_next <- t(params[[W_name]]) %*% dZ
    }
  }

  return(grads)
}

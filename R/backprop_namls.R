#' Backpropagation for NAMLS Model
#'
#' Computes gradients of the loss function with respect to all model parameters
#' in a NAMLS network using backpropagation. Supports dropout, ReLU activations,
#' and subnet-specific weight updates.
#'
#' @param X Numeric matrix of input features (predictors), with columns as samples
#'   and rows as features.
#' @param y Numeric vector of target values, same length as number of columns in `X`.
#' @param fwd List; result of a forward pass (`forward_namls()`), containing cached
#'   activations, outputs, and intermediate values.
#' @param params List of model parameters (weights and biases) for all subnetworks.
#' @param dropout_rate Numeric (0–1); dropout rate applied in hidden layers. Default is 0.
#'
#' @return A list of gradients (`grads`) for all model parameters, including
#'   weights, biases, and global coefficients (`beta_mu`, `beta_sigma`).
#'
#' @examples
#' \dontrun{
#' # Forward pass
#' fwd <- forward_namls(X, params)
#'
#' # Backpropagation
#' grads <- backprop_namls(X, y, fwd, params, dropout_rate = 0.2)
#' }
#'
#' @export

backprop_namls <- function(X, y, fwd, params, dropout_rate = 0) {

  cache <- fwd$cache
  n_features <- cache$n_features
  n_layers <- cache$n_layers
  batch_size <- ncol(X)

  grads <- list()

  # Output aus Forward
  mu <- cache$mu
  sigma <- cache$sigma
  sigma_raw <- cache$sigma_raw

  # Loss Gradienten
  dL_dmu <- -(y - mu) / (sigma^2)
  dL_dsigma <- 1/sigma - (y - mu)^2 / (sigma^3)
  # d/dx softplus(x) = d/dx log(1 + exp(x)) = exp(x)/(1 + exp(x)) = sigmoid(x)
  dL_dsigma_raw <- dL_dsigma * sigmoid(sigma_raw)

  grads$dbeta_mu <- sum(dL_dmu) / batch_size
  grads$dbeta_sigma <- sum(dL_dsigma_raw) / batch_size

  # Backprop durch jedes Subnetz
  for (j in 1:n_features) {
    # Cache für Subnetz j holen
    cache_j <- cache[[paste0("subnet", j)]]

    # Output Layer Gradienten
    delta <- rbind(dL_dmu, dL_dsigma_raw)

    for (l in n_layers:1) {
      W_name <- paste0("W", j, "_", l)
      b_name <- paste0("b", j, "_", l)
      dW_name <- paste0("dW", j, "_", l)
      db_name <- paste0("db", j, "_", l)

      if (l > 1) {
        A_prev <- cache_j[[paste0("A", l-1)]]
      } else {
        A_prev <- cache_j$A0  # Input
      }

      # Weight Gradienten berechnen
      grads[[dW_name]] <- delta %*% t(A_prev) / batch_size
      grads[[db_name]] <- matrix(rowSums(delta) / batch_size,
                                 nrow = nrow(delta), ncol = 1)

      # Backprop zu vorheriger Schicht
      if (l > 1) {
        delta <- t(params[[W_name]]) %*% delta

        # ReLU Gradient
        Z_prev <- cache_j[[paste0("Z", l-1)]]
        delta <- delta * (Z_prev > 0)

        # Dropout Gradient
        if (l == 2 && dropout_rate > 0) {
          if (!is.null(cache_j$dropout_mask)) {
            delta <- delta * cache_j$dropout_mask / (1 - dropout_rate)
          }
        }
      }
    }
  }

  return(grads)
}

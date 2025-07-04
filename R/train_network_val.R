#' Train a Single-Hidden-Layer Neural Network with Validation Loss Monitoring
#'
#' Trains a feed-forward neural network with one hidden layer using stochastic gradient descent,
#' and records both training and validation loss per epoch.
#'
#' @param train_loader A list of batches. Each batch must be a list containing:
#'   \code{$batch}: Input matrix of shape (observations × features),
#'   \code{$idx}: Integer indices corresponding to the observations in the original dataset.
#' @param targets Numeric vector of target values for all training observations.
#' @param dimensions A list specifying the network dimensions, e.g., as returned by \code{getLayerDimensions()}.
#' @param validation_X Numeric matrix of validation inputs (observations × features).
#' @param validation_y Numeric vector of target values for validation observations.
#' @param epochs Integer specifying the number of training epochs. Default is 100.
#' @param lr Numeric learning rate. Default is 0.01.
#'
#' @return A list containing:
#' \itemize{
#'   \item \code{params}: A list of trained network parameters (weights and biases).
#'   \item \code{train_loss}: Numeric vector of average training loss per epoch.
#'   \item \code{val_loss}: Numeric vector of validation loss per epoch.
#' }
#'
#' @details
#' For each epoch, the function iterates over all training batches, performs a forward pass,
#' computes the negative log-likelihood loss, backpropagates gradients, and updates parameters.
#' After all batches are processed, the model is evaluated once on the full validation set
#' to compute the validation loss. Both training and validation losses are recorded.
#'
#' @examples
#' # Example usage:
#' data <- matrix(rnorm(100 * 10), nrow = 100)
#' targets <- rnorm(100)
#' # Split your data here into train/validation
#' train_loader <- create_batches(data[1:80, ], batch_size = 20)
#' dims <- getLayerDimensions(data, out_dim = 2, hidden_neurons = 5)
#' validation_X <- data[81:100, ]
#' validation_y <- targets[81:100]
#' result <- train_network(train_loader, targets[1:80], dims, validation_X, validation_y,
#'                         epochs = 10, lr = 0.01)
#'
#' @export

train_network_val <- function(train_loader, targets, dimensions,
                              validation_X, validation_y,
                              epochs = 100, lr = 0.01) {

  params  <- init_params(dimensions)
  history_train <- numeric(epochs)
  history_val   <- numeric(epochs)

  for (e in seq_len(epochs)) {
    batch_losses <- numeric(length(train_loader))

    for (i in seq_along(train_loader)) {
      Xb <- train_loader[[i]]$batch
      yb <- targets[ train_loader[[i]]$idx ]

      fwd <- forward_onehidden(Xb, params)

      # Mean loss per batch
      batch_losses[i] <- neg_log_lik(
        yb,
        as.numeric(fwd$mu),
        as.numeric(fwd$log_sigma),
        reduction = "mean"
      )

      grads <- backprop_onehidden(Xb, yb, fwd, params)

      # Parameter update
      params$W1 <- params$W1 - lr * grads$dW1
      params$b1 <- params$b1 - lr * grads$db1
      params$W2 <- params$W2 - lr * grads$dW2
      params$b2 <- params$b2 - lr * grads$db2
    }

    # Mean training loss
    history[e] <- mean(batch_losses)

    #Validation loss
    fwd_val <- forward_onehidden(validation_X, params)
    history_val[e] <- neg_log_lik(
      validation_y,
      as.numeric(fwd_val$mu),
      as.numeric(fwd_val$log_sigma),
      reduction = "mean"
    )

    message(sprintf(
      "Epoch %3d/%d - Train Loss: %.6f  - Val Loss: %.6f",
      e, epochs, history_train[e], history_val[e]
    ))
  }

  list(
    params = params,
    train_loss = history_train,
    val_loss = history_val
  )
}

#' Train a Single-Hidden-Layer Neural Network
#'
#' Trains a feed-forward neural network with one hidden layer using stochastic gradient descent.
#'
#' @param train_loader A list of batches. Each batch must be a list containing:
#'   \code{$batch}: Input matrix of shape (features × batch size),
#'   \code{$idx}: Integer indices corresponding to the observations in the original dataset.
#' @param targets Numeric vector of target values for all observations.
#' @param dimensions A list specifying the network dimensions, e.g., as returned by \code{getLayerDimensions()}.
#' @param epochs Integer specifying the number of training epochs. Default is 100.
#' @param lr Numeric learning rate. Default is 0.01.
#'
#' @return A list containing:
#' \itemize{
#'   \item \code{params}: A list of trained network parameters (weights and biases).
#'   \item \code{history}: A numeric vector of average epoch losses.
#' }
#'
#' @details
#' For each epoch, this function iterates over all batches, performs a forward pass,
#' computes the negative log-likelihood loss, backpropagates gradients, and updates parameters.
#' The loss reported per epoch is the mean of the mean losses per batch.
#'
#' @examples
#' # Example usage:
#' data <- matrix(rnorm(100 * 10), nrow = 10)
#' loader <- DataLoader(data, batch_size = 20)
#' dims <- getLayerDimensions(data, out_dim = 2, hidden_neurons = 5)
#' result <- train_network(loader, rnorm(100), dims, epochs = 10, lr = 0.01)
#'
#' @export
train_network <- function(train_loader, targets, dimensions,
                          epochs = 100, lr = 0.01) {
  params  <- init_params(dimensions)
  history <- numeric(epochs)

  for (e in seq_len(epochs)) {
    batch_losses <- numeric(length(train_loader))

    for (i in seq_along(train_loader)) {
      Xb <- train_loader[[i]]$batch
      yb <- targets[ train_loader[[i]]$idx ]

      fwd <- forward_onehidden(Xb, params)
      # Mean loss per batch
      batch_losses[i] <- neg_log_lik(yb,
                                     as.numeric(fwd$mu),
                                     as.numeric(fwd$log_sigma),
                                     reduction = "mean") # mean oder raw?

      grads <- backprop_onehidden(Xb, yb, fwd, params)
      params$W1 <- params$W1 - lr * grads$dW1
      params$b1 <- params$b1 - lr * grads$db1
      params$W2 <- params$W2 - lr * grads$dW2
      params$b2 <- params$b2 - lr * grads$db2
    }

    # Mean epoch loss
    history[e] <- mean(batch_losses)
    message(sprintf("Epoch %3d/%d – Loss: %.6f",
                    e, epochs, history[e]))
  }

  list(params = params, history = history)
}

#' Train a Neural Network Model
#'
#' Trains a feedforward neural network using stochastic gradient descent (SGD)
#' or Adam optimization. Supports validation splits, early stopping, and
#' restoring best weights. Returns the trained model parameters and training
#' history.
#'
#' @param train_loader List of training batches, each containing a matrix `batch`
#'   and corresponding indices `idx`.
#' @param targets Numeric vector of target values for the full dataset.
#' @param val_split Optional validation set matrix (default: `NULL`).
#' @param hidden_neurons Integer vector; number of hidden neurons per layer
#'   (default: `c(50)`).
#' @param epochs Number of training epochs (default: `100`).
#' @param lr Learning rate (default: `0.01`).
#' @param optimizer Optimization algorithm, either `"sgd"` or `"adam"` (default: `"sgd"`).
#' @param beta1 Exponential decay rate for the first moment estimate (Adam only, default: `0.9`).
#' @param beta2 Exponential decay rate for the second moment estimate (Adam only, default: `0.999`).
#' @param eps Small constant for numerical stability in Adam optimizer (default: `1e-8`).
#' @param verbose Logical; if `TRUE` (default), prints training progress and
#'   summary information.
#' @param early_stopping Logical; if `TRUE` (default), enables early stopping
#'   based on validation loss.
#' @param es_patience Number of epochs with no improvement before stopping (default: `20`).
#' @param es_warmup Minimum number of epochs before early stopping can be triggered (default: `50`).
#' @param es_min_delta Minimum required improvement in validation loss to reset patience (default: `0`).
#' @param restore_best_weights Logical; if `TRUE` (default), restores model
#'   weights from the best validation epoch.
#'
#' @return A list of class `"NN"` containing:
#' \item{params}{Trained model parameters.}
#' \item{train_loss}{Vector of training loss values across epochs.}
#' \item{val_loss}{Vector of validation loss values (if validation split provided).}
#' \item{architecture}{Model architecture details.}
#' \item{epochs}{Number of training epochs performed.}
#' \item{lr}{Learning rate used in training.}
#' \item{optimizer}{Optimizer used (`"sgd"` or `"adam"`).}
#' \item{normalization}{Normalization parameters from the training data.}
#' \item{targets}{Original target values.}
#'
#' @examples
#' \dontrun{
#' # Example: training with validation split
#' model <- train.DNN(
#'   train_loader, targets,
#'   val_split = val_data,
#'   hidden_neurons = c(64, 32),
#'   epochs = 200, lr = 0.001,
#'   optimizer = "adam"
#' )
#' }
#'
#' @export

train.DNN <- function(
    train_loader, targets,
    val_split = NULL, hidden_neurons=c(50),
    epochs = 100, lr = 0.01,
    optimizer = c("sgd", "adam"),
    beta1 = 0.9, beta2 = 0.999, eps = 1e-8,
    verbose = TRUE,
    early_stopping = TRUE,
    es_patience = 20, es_warmup = 50, es_min_delta = 0,
    restore_best_weights = TRUE
) {

  #
  #if validation split was provided: transpose it and get respective targets
  if (!is.null(val_split)) {
    val_split <- t(val_split)
    val_targets <- targets[as.integer(rownames(t(val_split)))] #Transponieren nochmal checken!
  }
  else {
    val_targets <- NULL
  }

  optimizer <- match.arg(optimizer)

  dimensions <- getLayerDimensions(train_loader[[1]]$batch, hidden_neurons=hidden_neurons)

  params <- init_params(dimensions)
  arch <- attr(params, "architecture")
  n_layers <- arch$n_layers

  normalization_params <- train_loader[[1]]$normalization_params

  # Initialize Adam optimizer states if needed
  if (optimizer == "adam") {
    opt <- list()
    t_global <- 0

    # Initialize momentum and velocity for all parameters
    for (name in names(params)) {
      if (startsWith(name, "W") || startsWith(name, "b")) {
        opt[[paste0("m", name)]] <- matrix(0, nrow = nrow(params[[name]]),
                                           ncol = ncol(params[[name]]))
        opt[[paste0("v", name)]] <- matrix(0, nrow = nrow(params[[name]]),
                                           ncol = ncol(params[[name]]))
      }
    }
  }

  #Initialize parameters for early stopping
  es_wait <- 0
  best_params <- NULL
  best_val_loss <- Inf
  best_epoch <- NA_integer_



  history_train <- numeric(epochs)
  history_val <- if (!is.null(val_split)) numeric(epochs) else NULL

  # Training loop
  for (e in seq_len(epochs)) {
    batch_losses <- numeric(length(train_loader))
    batch_sizes <- numeric(length(train_loader))

    for (i in seq_along(train_loader)) {
      Xb <- train_loader[[i]]$batch
      yb <- targets[train_loader[[i]]$idx]
      batch_sizes[i] <- length(yb)

      # Forward pass
      fwd <- forward(Xb, params)

      # Compute loss
      batch_losses[i] <- neg_log_lik(
        yb, as.numeric(fwd$mu), as.numeric(fwd$log_sigma),
        reduction = "mean"
      )

      # Backward pass
      grads <- backprop(Xb, yb, fwd, params)

      # Update parameters
      if (optimizer == "sgd") {
        for (l in 1:(n_layers + 1)) {
          W_name <- paste0("W", l)
          b_name <- paste0("b", l)
          dW_name <- paste0("dW", l)
          db_name <- paste0("db", l)

          params[[W_name]] <- params[[W_name]] - lr * grads[[dW_name]]
          params[[b_name]] <- params[[b_name]] - lr * grads[[db_name]]
        }
      } else {  # Adam
        t_global <- t_global + 1

        for (l in 1:(n_layers + 1)) {
          W_name <- paste0("W", l)
          b_name <- paste0("b", l)
          dW_name <- paste0("dW", l)
          db_name <- paste0("db", l)

          # Update W
          mW_name <- paste0("mW", l)
          vW_name <- paste0("vW", l)
          tmp <- update_adam(opt[[mW_name]], opt[[vW_name]], grads[[dW_name]],
                             beta1, beta2, t_global, lr, eps)
          opt[[mW_name]] <- tmp$m
          opt[[vW_name]] <- tmp$v
          params[[W_name]] <- params[[W_name]] - tmp$delta

          # Update b
          mb_name <- paste0("mb", l)
          vb_name <- paste0("vb", l)
          tmp <- update_adam(opt[[mb_name]], opt[[vb_name]], grads[[db_name]],
                             beta1, beta2, t_global, lr, eps)
          opt[[mb_name]] <- tmp$m
          opt[[vb_name]] <- tmp$v
          params[[b_name]] <- params[[b_name]] - tmp$delta
        }
      }
    }

    # Log losses
    history_train[e] <- weighted.mean(batch_losses, batch_sizes)

    # Calculation of validation loss
    if (!is.null(val_split)) {
      fwd_val <- forward(val_split, params)
      history_val[e] <- neg_log_lik(
        val_targets,
        as.numeric(fwd_val$mu),
        as.numeric(fwd_val$log_sigma),
        reduction = "mean"
      )

      #If validation loss has improved, reset counter for early stopping
      prev_best <- best_val_loss
      improved  <- (history_val[e] < (prev_best - es_min_delta))

      if (improved) {
        best_val_loss <- history_val[e]
        es_wait <- 0
        best_epoch <- e
        if (restore_best_weights) best_params <- unserialize(serialize(params, NULL))
      } else {
        es_wait <- es_wait + 1
      }

      message(sprintf(
        "Epoch %3d/%d - Train loss: %.6f | Validation loss: %.6f",
        e, epochs, history_train[e], history_val[e]
      ))
    } else {
      message(sprintf(
        "Epoch %3d/%d - Loss: %.6f",
        e, epochs, history_train[e]
      ))
    }


    # Early Stopping
    # If warm-up period is over and no val improvement over defined epoch nr: stop training
    if (early_stopping){
    if (e >= es_warmup && es_wait >= es_patience) {
      if (verbose) {
        cat(sprintf("Early stopping after %d epochs without improvement of validation loss. \nBest validation loss: %.6f @ epoch %d\n",
                    es_wait, best_val_loss, best_epoch))
      }
      #If requested, reset params to those that produce best val loss
      if (restore_best_weights && !is.null(best_params)) {
        params <- best_params
      }
      history_train <- history_train[1:e]
      history_val   <- history_val[1:e]
      break
      }
    }

  }

  if (verbose) {
    cat("\n===============================================\n")
    cat("Training completed!\n")


  # Display architecture after last epoch
  arch_str <- paste(c(arch$n_x, arch$n_h, arch$n_y), collapse = " -> ")
  message(sprintf("Trained network with architecture: %s", arch_str))

  cat("===============================================\n")
  }
  # Preserve architecture in params
  attr(params, "architecture") <- arch

  out <- list(
    params = params,
    train_loss = history_train,
    architecture = arch,
    epochs = epochs,
    lr = lr,
    optimizer = optimizer,
    normalization = normalization_params,
    targets = targets
  )
  if (!is.null(history_val)) out$val_loss <- history_val

  class(out) <- c("NN", class(out))
  return(out)
}

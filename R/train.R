#' Train a Multi-Layer Neural Network (SGD oder Adam, optional mit Validation)
#'
#' @param train_loader  Liste von Batches (jeweils $batch und $idx).
#' @param targets       Zielwerte für alle Beobachtungen.
#' @param hidden_neurons Vector of neurons per hidden layer, e.g., c(50, 30, 20) chronologically ordered flowing from input to output, i.e. left to right
#' @param epochs        Anzahl Epochen (Default 100).
#' @param lr            Lernrate (Default 0.01).
#' @param optimizer     Optimizer, entweder "sgd" oder "adam" (Default "sgd").
#' @param beta1         Adam β1 (nur wenn optimizer = "adam", Default 0.9).
#' @param beta2         Adam β2 (nur wenn optimizer = "adam", Default 0.999).
#' @param eps           Adam ε (nur wenn optimizer = "adam", Default 1e-8).
#' @param val_split  Optional: Matrix für Validierungs-Inputs.
#' @param val_targets  Optional: Vektor für Validierungs-Ziele.
#'
#' @return Liste mit
#' \item{params}{Gelernte Parameter}
#' \item{train_loss}{Vektor der Trainingsverluste pro Epoche}
#' \item{val_loss}{Optional: Vektor der Validierungsverluste pro Epoche}
#' @export
train <- function(
    train_loader, targets,
    val_split = NULL, val_targets = NULL, hidden_neurons,
    epochs = 100, lr = 0.01,
    optimizer = c("sgd", "adam"),
    beta1 = 0.9, beta2 = 0.999, eps = 1e-8,
    normalization_params = NULL # was macht diese Variable??
) {
  optimizer <- match.arg(optimizer)

  dimensions <- getLayerDimensions(train_loader[[1]]$batch, hidden_neurons=hidden_neurons)

  params <- init_params(dimensions)
  arch <- attr(params, "architecture")
  n_layers <- arch$n_layers

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

    if (!is.null(val_split)) {
      fwd_val <- forward(val_split, params)
      history_val[e] <- neg_log_lik(
        val_targets,
        as.numeric(fwd_val$mu),
        as.numeric(fwd_val$log_sigma),
        reduction = "mean"
      )
      message(sprintf(
        "Epoch %3d/%d – Train: %.6f | Val: %.6f",
        e, epochs, history_train[e], history_val[e]
      ))
    } else {
      message(sprintf(
        "Epoch %3d/%d – Loss: %.6f",
        e, epochs, history_train[e]
      ))
    }
  }

  # Display architecture after last epoch
  arch_str <- paste(c(arch$n_x, arch$n_h, arch$n_y), collapse = " -> ")
  message(sprintf("Trained network with architecture: %s", arch_str))

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

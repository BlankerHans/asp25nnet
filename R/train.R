#' Train a Single-Hidden-Layer Neural Network (SGD oder Adam, optional mit Validation)
#'
#' @param train_loader  Liste von Batches (jeweils $batch und $idx).
#' @param targets       Zielwerte für alle Beobachtungen.
#' @param dimensions    Netzwerk-Dimensionen (aus getLayerDimensions()).
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
    train_loader, targets, dimensions, val_split = NULL, val_targets = NULL,
    epochs = 100, lr = 0.01,
    optimizer = c("sgd", "adam"),
    beta1 = 0.9, beta2 = 0.999, eps = 1e-8
) {
  optimizer <- match.arg(optimizer)
  params <- init_params(dimensions)

  if (optimizer == "adam") {
    opt <- list(
      mW1 = matrix(0, dimensions$n_h, dimensions$n_x),
      vW1 = matrix(0, dimensions$n_h, dimensions$n_x),
      mb1 = matrix(0, dimensions$n_h, 1),
      vb1 = matrix(0, dimensions$n_h, 1),
      mW2 = matrix(0, dimensions$n_y, dimensions$n_h),
      vW2 = matrix(0, dimensions$n_y, dimensions$n_h),
      mb2 = matrix(0, dimensions$n_y, 1),
      vb2 = matrix(0, dimensions$n_y, 1)
    )
    t_global <- 0
  }

  history_train <- numeric(epochs)
  history_val   <- if (!is.null(val_split)) numeric(epochs) else NULL

  # Trainingsloop
  for (e in seq_len(epochs)) {
    batch_losses <- numeric(length(train_loader))
    batch_sizes <- numeric(length(train_loader))

    for (i in seq_along(train_loader)) {
      Xb  <- train_loader[[i]]$batch
      yb  <- targets[train_loader[[i]]$idx]
      batch_sizes[i] <- length(yb)
      fwd <- forward_onehidden(Xb, params)

      # Loss
      batch_losses[i] <- neg_log_lik(
        yb, as.numeric(fwd$mu), as.numeric(fwd$log_sigma),
        reduction = "mean"
      ) # mean oder raw?

      # Gradienten
      grads <- backprop_onehidden(Xb, yb, fwd, params)

      if (optimizer == "sgd") {
        # einfacher Gradientenschritt
        params$W1 <- params$W1 - lr * grads$dW1
        params$b1 <- params$b1 - lr * grads$db1
        params$W2 <- params$W2 - lr * grads$dW2
        params$b2 <- params$b2 - lr * grads$db2

      } else {
        # Adam-Update
        t_global <- t_global + 1



        # W1
        tmp  <- update_adam(opt$mW1, opt$vW1, grads$dW1, beta1, beta2, t_global, lr, eps)
        opt$mW1 <- tmp$m; opt$vW1 <- tmp$v
        params$W1 <- params$W1 - tmp$delta

        # b1
        tmp  <- update_adam(opt$mb1, opt$vb1, grads$db1, beta1, beta2, t_global, lr, eps)
        opt$mb1 <- tmp$m; opt$vb1 <- tmp$v
        params$b1 <- params$b1 - tmp$delta

        # W2
        tmp  <- update_adam(opt$mW2, opt$vW2, grads$dW2, beta1, beta2, t_global, lr, eps)
        opt$mW2 <- tmp$m; opt$vW2 <- tmp$v
        params$W2 <- params$W2 - tmp$delta

        # b2
        tmp  <- update_adam(opt$mb2, opt$vb2, grads$db2, beta1, beta2, t_global, lr, eps)
        opt$mb2 <- tmp$m; opt$vb2 <- tmp$v
        params$b2 <- params$b2 - tmp$delta
      }
    }

    # Loss‐Logging
    history_train[e] <- weighted.mean(batch_losses, batch_sizes)  # wir berechnen mean of means
    if (!is.null(val_split)) {
      fwd_val <- forward_onehidden(val_split, params)
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

  out <- list(
    params = params,
    train_loss = history_train,
    epochs = epochs,
    lr = lr,
    optimizer = optimizer
  )
  if (!is.null(history_val)) out$val_loss <- history_val
  invisible(out)
}

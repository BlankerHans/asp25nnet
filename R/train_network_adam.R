train_network_val_adam <- function(
    train_loader, targets, dimensions,
    validation_X, validation_y,
    epochs = 100, lr = 0.001,
    beta1 = 0.9, beta2 = 0.999, eps = 1e-8
) {
  # 1) Parameter und Adam-State initialisieren
  params <- init_params(dimensions)
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
  history_train <- numeric(epochs)
  history_val   <- numeric(epochs)

  # 2) Epoch‐Loop
  t_global <- 0
  for (e in seq_len(epochs)) {
    batch_losses <- numeric(length(train_loader))

    # 2a) Batch‐Loop mit Adam‐Updates
    for (i in seq_along(train_loader)) {
      t_global <- t_global + 1
      Xb <- train_loader[[i]]$batch
      yb <- targets[train_loader[[i]]$idx]

      fwd  <- forward_onehidden(Xb, params)
      loss <- neg_log_lik(yb,
                          as.numeric(fwd$mu),
                          as.numeric(fwd$log_sigma),
                          reduction = "mean")
      batch_losses[i] <- loss

      grads <- backprop_onehidden(Xb, yb, fwd, params)

      # --- W1 ---
      opt$mW1 <- beta1*opt$mW1 + (1-beta1)*grads$dW1
      opt$vW1 <- beta2*opt$vW1 + (1-beta2)*(grads$dW1^2)
      mW1_hat <- opt$mW1 / (1 - beta1^t_global)
      vW1_hat <- opt$vW1 / (1 - beta2^t_global)
      params$W1 <- params$W1 - lr * mW1_hat / (sqrt(vW1_hat) + eps)

      # --- b1 ---
      opt$mb1 <- beta1*opt$mb1 + (1-beta1)*grads$db1
      opt$vb1 <- beta2*opt$vb1 + (1-beta2)*(grads$db1^2)
      mb1_hat <- opt$mb1 / (1 - beta1^t_global)
      vb1_hat <- opt$vb1 / (1 - beta2^t_global)
      params$b1 <- params$b1 - lr * mb1_hat / (sqrt(vb1_hat) + eps)

      # --- W2 ---
      opt$mW2 <- beta1*opt$mW2 + (1-beta1)*grads$dW2
      opt$vW2 <- beta2*opt$vW2 + (1-beta2)*(grads$dW2^2)
      mW2_hat <- opt$mW2 / (1 - beta1^t_global)
      vW2_hat <- opt$vW2 / (1 - beta2^t_global)
      params$W2 <- params$W2 - lr * mW2_hat / (sqrt(vW2_hat) + eps)

      # --- b2 ---
      opt$mb2 <- beta1*opt$mb2 + (1-beta1)*grads$db2
      opt$vb2 <- beta2*opt$vb2 + (1-beta2)*(grads$db2^2)
      mb2_hat <- opt$mb2 / (1 - beta1^t_global)
      vb2_hat <- opt$vb2 / (1 - beta2^t_global)
      params$b2 <- params$b2 - lr * mb2_hat / (sqrt(vb2_hat) + eps)
    }

    # 2b) Loss‐Logging
    history_train[e] <- mean(batch_losses)
    fwd_val <- forward_onehidden(validation_X, params)
    history_val[e] <- neg_log_lik(
      validation_y,
      as.numeric(fwd_val$mu),
      as.numeric(fwd_val$log_sigma),
      reduction = "mean"
    )

    message(sprintf(
      "Epoch %3d/%d – Train: %.6f  | Val: %.6f",
      e, epochs, history_train[e], history_val[e]
    ))
  }

  list(
    params      = params,
    train_loss  = history_train,
    val_loss    = history_val
  )
}

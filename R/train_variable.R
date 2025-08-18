train_variable <- function(
    train_loader, targets, dimensions,
    val_split = NULL, val_targets = NULL,
    epochs = 100, lr = 0.01,
    optimizer = c("sgd", "adam"),
    beta1 = 0.9, beta2 = 0.999, eps = 1e-8
) {
  optimizer <- match.arg(optimizer)
  params <- init_params_variable(dimensions)
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
      fwd <- forward_variable(Xb, params)

      # Compute loss
      batch_losses[i] <- neg_log_lik(
        yb, as.numeric(fwd$mu), as.numeric(fwd$log_sigma),
        reduction = "mean"
      )

      # Backward pass
      grads <- backprop_variable(Xb, yb, fwd, params)

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
      fwd_val <- forward_variable(val_split, params)
      history_val[e] <- neg_log_lik(
        val_targets,
        as.numeric(fwd_val$mu),
        as.numeric(fwd_val$log_sigma),
        reduction = "mean"
      )

      # Display architecture in first epoch
      if (e == 1) {
        arch_str <- paste(c(arch$n_x, arch$n_h, arch$n_y), collapse = " -> ")
        message(sprintf("Training network with architecture: %s", arch_str))
      }

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

  # Preserve architecture in params
  attr(params, "architecture") <- arch

  out <- list(
    params = params,
    train_loss = history_train,
    architecture = arch
  )
  if (!is.null(history_val)) out$val_loss <- history_val

  return(out)
}

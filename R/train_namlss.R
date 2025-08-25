#' Train NAMLSS
#'
#' @param train_loader DataLoader mit Trainingsdaten
#' @param targets Alle Zielwerte
#' @param n_features Anzahl Features
#' @param hidden_neurons Hidden Layer Größen
#' @param val_split Validierungsdaten (optional)
#' @param val_targets Validierungsziele (optional)
#' @param epochs Anzahl Epochen
#' @param lr Lernrate
#' @param optimizer "sgd" oder "adam"
#' @param dropout_rate Dropout Rate (Paper: 0.5 für kleine Datensätze)
#' @param beta1 Adam Parameter
#' @param beta2 Adam Parameter
#' @param eps Adam Parameter
#' @param lr_decay Learning rate decay factor
#' @param lr_patience Epochs to wait before decay
#' @param verbose Fortschritt anzeigen
#' @return Trainiertes Modell
#' @export
train_namlss <- function(train_loader, targets, n_features,
                         hidden_neurons = c(250, 50, 25),
                         val_split = NULL, val_targets = NULL,
                         epochs = 100, lr = 0.01,
                         optimizer = c("adam", "sgd"),
                         dropout_rate = 0.5,
                         beta1 = 0.9, beta2 = 0.999, eps = 1e-8,
                         lr_decay = 0.95, lr_patience = 10,
                         verbose = TRUE) {

  optimizer <- match.arg(optimizer)

  # Initialisiere Parameter
  params <- init_namlss_params(n_features, hidden_neurons, mean(targets),
                               sd(targets), sd(targets))
  arch <- attr(params, "architecture")

  # Adam Optimizer Setup
  if (optimizer == "adam") {
    opt <- list()
    t_global <- 0

    # Initialisiere Momentum und Velocity für alle Parameter
    for (name in names(params)) {
      if (startsWith(name, "W") || startsWith(name, "b") ||
          name %in% c("beta_mu", "beta_sigma")) {
        if (is.matrix(params[[name]])) {
          opt[[paste0("m_", name)]] <- matrix(0, nrow = nrow(params[[name]]),
                                              ncol = ncol(params[[name]]))
          opt[[paste0("v_", name)]] <- matrix(0, nrow = nrow(params[[name]]),
                                              ncol = ncol(params[[name]]))
        } else {
          # Für skalare Parameter
          opt[[paste0("m_", name)]] <- 0
          opt[[paste0("v_", name)]] <- 0
        }
      }
    }
  }

  # Learning Rate Scheduler Setup
  lr_current <- lr
  lr_counter <- 0
  best_val_loss <- Inf

  # Training History
  history_train <- numeric(epochs)
  history_val <- if (!is.null(val_split)) numeric(epochs) else NULL

  # Training Loop
  for (e in 1:epochs) {
    epoch_start <- Sys.time()
    batch_losses <- numeric(length(train_loader))
    batch_sizes <- numeric(length(train_loader))

    # Mini-batch Training
    for (i in seq_along(train_loader)) {
      Xb <- train_loader[[i]]$batch
      yb <- targets[train_loader[[i]]$idx]
      batch_sizes[i] <- length(yb)

      # Forward Pass mit Cache
      fwd <- forward_namlss(Xb, params, dropout_rate, training = TRUE)

      # Loss berechnen
      batch_losses[i] <- neg_log_lik(
        yb, fwd$mu, fwd$log_sigma,
        reduction = "mean"
      )

      # Backpropagation mit Cache
      grads <- backprop_namlss(Xb, yb, fwd, params, dropout_rate)

      # Parameter Update
      if (optimizer == "sgd") {
        # SGD Update
        for (j in 1:n_features) {
          for (l in 1:arch$n_layers) {
            # Weights
            W_name <- paste0("W", j, "_", l)
            dW_name <- paste0("dW", j, "_", l)
            if (dW_name %in% names(grads)) {
              params[[W_name]] <- params[[W_name]] - lr_current * grads[[dW_name]]
            }

            # Bias
            b_name <- paste0("b", j, "_", l)
            db_name <- paste0("db", j, "_", l)
            if (db_name %in% names(grads)) {
              params[[b_name]] <- params[[b_name]] - lr_current * grads[[db_name]]
            }
          }
        }

        # Globale Bias Updates
        params$beta_mu <- params$beta_mu - lr_current * grads$dbeta_mu
        params$beta_sigma <- params$beta_sigma - lr_current * grads$dbeta_sigma

      } else {  # Adam
        t_global <- t_global + 1

        # Update alle Parameter mit Adam
        for (j in 1:n_features) {
          for (l in 1:arch$n_layers) {
            # Weights
            W_name <- paste0("W", j, "_", l)
            dW_name <- paste0("dW", j, "_", l)

            if (dW_name %in% names(grads)) {
              mW_name <- paste0("m_", W_name)
              vW_name <- paste0("v_", W_name)

              # Adam update
              opt[[mW_name]] <- beta1 * opt[[mW_name]] + (1 - beta1) * grads[[dW_name]]
              opt[[vW_name]] <- beta2 * opt[[vW_name]] + (1 - beta2) * grads[[dW_name]]^2

              m_hat <- opt[[mW_name]] / (1 - beta1^t_global)
              v_hat <- opt[[vW_name]] / (1 - beta2^t_global)

              params[[W_name]] <- params[[W_name]] - lr_current * m_hat / (sqrt(v_hat) + eps)
            }

            # Bias
            b_name <- paste0("b", j, "_", l)
            db_name <- paste0("db", j, "_", l)

            if (db_name %in% names(grads)) {
              mb_name <- paste0("m_", b_name)
              vb_name <- paste0("v_", b_name)

              opt[[mb_name]] <- beta1 * opt[[mb_name]] + (1 - beta1) * grads[[db_name]]
              opt[[vb_name]] <- beta2 * opt[[vb_name]] + (1 - beta2) * grads[[db_name]]^2

              m_hat <- opt[[mb_name]] / (1 - beta1^t_global)
              v_hat <- opt[[vb_name]] / (1 - beta2^t_global)

              params[[b_name]] <- params[[b_name]] - lr_current * m_hat / (sqrt(v_hat) + eps)
            }
          }
        }

        # Globale Bias Updates mit Adam
        # Beta_mu
        opt$m_beta_mu <- beta1 * opt$m_beta_mu + (1 - beta1) * grads$dbeta_mu
        opt$v_beta_mu <- beta2 * opt$v_beta_mu + (1 - beta2) * grads$dbeta_mu^2
        m_hat <- opt$m_beta_mu / (1 - beta1^t_global)
        v_hat <- opt$v_beta_mu / (1 - beta2^t_global)
        params$beta_mu <- params$beta_mu - lr_current * m_hat / (sqrt(v_hat) + eps)

        # Beta_sigma
        opt$m_beta_sigma <- beta1 * opt$m_beta_sigma + (1 - beta1) * grads$dbeta_sigma
        opt$v_beta_sigma <- beta2 * opt$v_beta_sigma + (1 - beta2) * grads$dbeta_sigma^2
        m_hat <- opt$m_beta_sigma / (1 - beta1^t_global)
        v_hat <- opt$v_beta_sigma / (1 - beta2^t_global)
        params$beta_sigma <- params$beta_sigma - lr_current * m_hat / (sqrt(v_hat) + eps)
      }
    }

    # Epoche Loss
    history_train[e] <- weighted.mean(batch_losses, batch_sizes)

    # Validierung
    if (!is.null(val_split)) {
      fwd_val <- forward_namlss(val_split, params,
                                           dropout_rate = 0, training = FALSE)
      history_val[e] <- neg_log_lik(
        val_targets, fwd_val$mu, fwd_val$log_sigma,
        reduction = "mean"
      )

      # Learning Rate Scheduler
      if (history_val[e] < best_val_loss) {
        best_val_loss <- history_val[e]
        lr_counter <- 0
      } else {
        lr_counter <- lr_counter + 1
        if (lr_counter >= lr_patience) {
          lr_current <- lr_current * lr_decay
          lr_counter <- 0
          if (verbose) {
            cat(sprintf("  Learning rate reduced to %.6f\n", lr_current))
          }
        }
      }

      # Ausgabe
      if (verbose && (e %% 10 == 0 || e == 1 || e == epochs)) {
        epoch_time <- as.numeric(Sys.time() - epoch_start, units = "secs")
        cat(sprintf("Epoch %3d/%d - Train: %.6f | Val: %.6f | Time: %.2fs | LR: %.6f\n",
                    e, epochs, history_train[e], history_val[e], epoch_time, lr_current))
      }
    } else {
      # Ohne Validierung
      if (verbose && (e %% 10 == 0 || e == 1 || e == epochs)) {
        epoch_time <- as.numeric(Sys.time() - epoch_start, units = "secs")
        cat(sprintf("Epoch %3d/%d - Loss: %.6f | Time: %.2fs\n",
                    e, epochs, history_train[e], epoch_time))
      }
    }

    # Early Stopping Check
    if (!is.null(val_split) && e > 50) {
      # Prüfe ob Validierung sich verbessert
      if (e > 20 && all(diff(tail(history_val, 20)) > 0)) {
        if (verbose) {
          cat("Early stopping: Validation loss not improving for 20 epochs\n")
        }
        history_train <- history_train[1:e]
        history_val <- history_val[1:e]
        break
      }
    }
  }

  # Finales Modell Info
  if (verbose) {
    cat("\n===============================================\n")
    cat("Training completed!\n")
    cat("===============================================\n")
    arch_str <- paste(c(1, arch$n_h, 2), collapse = " -> ")
    cat(sprintf("NAMLSS with %d feature networks\n", n_features))
    cat(sprintf("Each network: %s\n", arch_str))

    # Parameter Anzahl
    n_params <- 0
    for (name in names(params)) {
      if (startsWith(name, "W") || startsWith(name, "b")) {
        n_params <- n_params + length(params[[name]])
      }
    }
    n_params <- n_params + 2  # beta_mu, beta_sigma

    cat(sprintf("Total parameters: %d\n", n_params))
    cat(sprintf("Final train loss: %.6f\n", tail(history_train, 1)))
    if (!is.null(history_val)) {
      cat(sprintf("Final val loss: %.6f\n", tail(history_val, 1)))
      cat(sprintf("Best val loss: %.6f\n", min(history_val)))
    }
    cat("===============================================\n")
  }

  # Model zusammenstellen
  model <- list(
    params = params,
    train_loss = history_train,
    val_loss = history_val,
    architecture = arch,
    n_features = n_features,
    epochs = length(history_train),
    lr = lr,
    final_lr = lr_current,
    optimizer = optimizer,
    dropout_rate = dropout_rate,
    best_val_loss = if (!is.null(val_split)) min(history_val) else NULL
  )

  class(model) <- c("NAMLSS", class(model))
  attr(model$params, "architecture") <- arch

  return(model)
}

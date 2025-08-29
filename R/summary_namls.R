#' Summary of a Trained NAMLS Model
#'
#' Provides a summary of a trained NAMLS model, including its architecture,
#' training setup, training results, and visualizations of model predictions.
#' Depending on the number of input features, it produces loss curves,
#' 1D fits with confidence intervals, or partial effect plots for multiple features.
#'
#' @param object Trained NAMLS model object containing parameters, optimizer, learning rate,
#'   epochs, losses, and training information.
#' @param data Data frame used for model evaluation, including predictors and target.
#' @param target_col Character string with the name of the target variable.
#' @param pm1_scaler List containing feature scaling parameters from pm1_scaler() function,
#'   with elements 'a' (min values), 'b' (max values), and 'eps'. If NULL, no feature
#'   scaling is applied.
#' @param target_mean Numeric; mean of target variable from training set used for
#'   normalization. If NULL, no target denormalization is performed.
#' @param target_sd Numeric; standard deviation of target variable from training set
#'   used for normalization. If NULL, no target denormalization is performed.
#' @param show_plot Logical; if TRUE, plots of losses and predictions are shown. Default is TRUE.
#' @param yscale Character; determines y-axis scaling for loss plots:
#'   \code{"auto"} (linear), \code{"log"} (logarithmic), \code{"robust"} (capped at quantile).
#' @param cap_quantile Numeric (0-1); quantile for robust loss capping. Default is 0.99.
#' @param drop_first Integer; number of initial epochs to exclude from loss plots. Default is 0.
#' @param feature_plots Logical; if TRUE and more than one feature is present, partial effect
#'   plots are generated. Default is TRUE.
#' @param max_features Integer; maximum number of features to display in partial effect plots.
#'   Default is 6.
#' @param ci_z Numeric; z-value used for confidence interval calculation.
#'   Default is 1.96 (approximately 95% CI).
#'
#' @return Invisibly returns the input \code{object} after printing a summary and
#'   optionally plotting results.
#'
#' @examples
#' \dontrun{
#' # Train model
#' model <- train_namls(...)
#'
#' # Get preprocessing parameters from training
#' pm1 <- pm1_scaler(train_data)
#' norm_targets <- normalize_targets(train_targets)
#'
#' # Summary with preprocessing info
#' summary.NAMLS(
#'   object = model,
#'   data = test_data,
#'   target_col = "y",
#'   pm1_scaler = pm1,
#'   target_mean = norm_targets$mean,
#'   target_sd = norm_targets$sd
#' )
#' }
#'
#' @export
summary.NAMLS <- function(object,
                          data,
                          target_col,
                          pm1_scaler = NULL,
                          target_mean = NULL,
                          target_sd = NULL,
                          show_plot = TRUE,
                          yscale = c("auto", "log", "robust"),
                          cap_quantile = 0.99,
                          drop_first = 0,
                          feature_plots = TRUE,
                          max_features = 6,
                          ci_z = 1.96) {

  yscale <- match.arg(yscale)

  # ---- Helper Functions ----
  Softplus_ <- function(z) log1p(exp(-abs(z))) + pmax(z, 0)

  plot_losses_ <- function(train_loss, val_loss = NULL, yscale = "auto",
                           cap_quantile = 0.99, drop_first = 0) {
    epochs <- seq_along(train_loss)
    if (drop_first > 0 && drop_first < length(epochs)) {
      epochs <- epochs[-(1:drop_first)]
      train_loss <- train_loss[-(1:drop_first)]
      if (!is.null(val_loss)) val_loss <- val_loss[-(1:drop_first)]
    }

    y_min <- min(train_loss, val_loss, na.rm = TRUE)
    y_max <- max(train_loss, val_loss, na.rm = TRUE)

    if (yscale == "robust" && length(train_loss) > 10) {
      cap_val <- quantile(c(train_loss, val_loss), cap_quantile, na.rm = TRUE)
      train_loss[train_loss > cap_val] <- cap_val
      if (!is.null(val_loss)) val_loss[val_loss > cap_val] <- cap_val
      y_max <- cap_val
    }

    plot(epochs, train_loss, type = "l", col = "blue", lwd = 2,
         ylim = c(y_min, y_max),
         xlab = "Epoch", ylab = "Loss",
         main = "Training Progress",
         log = if (yscale == "log") "y" else "")

    if (!is.null(val_loss)) {
      lines(epochs, val_loss, col = "red", lwd = 2)
      legend("topright", legend = c("Train", "Validation"),
             col = c("blue", "red"), lwd = 2, bty = "n")
    }
  }

  arch_print_ <- function(obj, pm1_scaler, target_mean, target_sd) {
    if (!is.null(obj$n_features)) {
      cat("Number of features:      ", obj$n_features, "\n", sep="")
    }
    if (!is.null(obj$architecture$layer_sizes)) {
      ls <- obj$architecture$layer_sizes
      cat("Subnet architecture:     ", paste(ls, collapse=" -> "), "\n", sep="")
    } else if (!is.null(obj$architecture$n_h)) {
      cat("Subnet architecture:     ",
          paste(c(1, obj$architecture$n_h, 2), collapse=" -> "), "\n", sep="")
    }
    cat("Optimizer:               ", obj$optimizer, "\n", sep="")
    cat("Loss function:           Negative Log-Likelihood\n")
    cat("Learning rate (start):   ", obj$lr, "\n", sep="")
    if (!is.null(obj$final_lr)) {
      cat("Learning rate (final):   ", obj$final_lr, "\n", sep="")
    }
    cat("Trained epochs:          ", length(obj$train_loss), "\n", sep="")
    cat(sprintf("Final training loss:     %.6f\n", tail(obj$train_loss, 1)))
    if (!is.null(obj$val_loss)) {
      cat(sprintf("Final validation loss:   %.6f\n", tail(obj$val_loss, 1)))
    }
    if (!is.null(obj$best_val_loss)) {
      cat(sprintf("Best validation loss:    %.6f\n", obj$best_val_loss))
    }

    # Display preprocessing information
    cat("\nPreprocessing:\n")
    if (!is.null(pm1_scaler)) {
      cat("  Features:              [-1, 1] scaling\n")
    } else {
      cat("  Features:              None (raw data)\n")
    }

    if (!is.null(target_mean) && !is.null(target_sd)) {
      cat("  Targets:               Standard normalization\n")
      cat(sprintf("    - mean: %.4f\n", target_mean))
      cat(sprintf("    - sd:   %.4f\n", target_sd))
    } else {
      cat("  Targets:               None (raw data)\n")
    }
  }

  # ---- Main Summary Output ----
  cat("==============================\n")
  cat("    NAMLS Model Summary\n")
  cat("==============================\n\n")
  arch_print_(object, pm1_scaler, target_mean, target_sd)

  # ---- (1) Loss Plot ----
  if (isTRUE(show_plot)) {
    cat("\n(1) Loss Curves\n")
    cat("------------------------------\n")
    plot_losses_(object$train_loss, object$val_loss, yscale, cap_quantile, drop_first)
  }

  if (!isTRUE(show_plot)) return(invisible(object))

  # ---- Data Preparation ----
  x_cols <- setdiff(names(data), target_col)
  if (length(x_cols) == 0) {
    warning("No feature columns found in data.")
    return(invisible(object))
  }

  X_raw <- as.matrix(data[, x_cols, drop = FALSE])

  # Apply feature preprocessing if pm1_scaler is provided
  if (!is.null(pm1_scaler)) {
    X_used <- transform_pm1(X_raw, pm1_scaler, clip = TRUE)
  } else {
    # No feature preprocessing
    X_used <- X_raw
  }

  X_t <- t(as.matrix(X_used))
  p <- ncol(X_raw)

  # Check feature dimensions
  if (!is.null(object$n_features) && p != object$n_features) {
    stop(sprintf("Number of features in 'data' (%d) does not match object$n_features (%d).",
                 p, object$n_features))
  }

  # ---- Forward Pass ----
  fwd <- forward_namls(X_t, object$params, dropout_rate = 0, training = FALSE)
  mu_pred <- as.numeric(fwd$mu)
  sigma_pred <- as.numeric(fwd$sigma)

  # ---- Denormalize predictions if target normalization parameters are provided ----
  if (!is.null(target_mean) && !is.null(target_sd)) {
    # Transform predictions back to original scale
    mu_original <- target_mean + target_sd * mu_pred
    sigma_original <- target_sd * sigma_pred

    # Use original scale for plotting
    mu_plot <- mu_original
    sigma_plot <- sigma_original
    y_plot <- data[[target_col]]  # Original scale targets

    cat("\nNote: Predictions denormalized to original scale.\n")
  } else {
    # No target denormalization
    mu_plot <- mu_pred
    sigma_plot <- sigma_pred
    y_plot <- data[[target_col]]
  }

  # ---- (2) Single Feature Plot ----
  if (p == 1) {
    cat("\n(2) Prediction Plot (1 Feature)\n")
    cat("------------------------------\n")

    x <- X_raw[, 1]
    ord <- order(x)
    x_s <- x[ord]
    y_s <- y_plot[ord]
    mu_s <- mu_plot[ord]
    sig_s <- sigma_plot[ord]

    upper <- mu_s + ci_z * sig_s
    lower <- mu_s - ci_z * sig_s

    graphics::par(mfrow = c(1,1))
    graphics::plot(x_s, y_s, pch = 16, cex = 0.6,
                   xlab = x_cols[1], ylab = target_col,
                   main = sprintf("NAMLS Fit (μ ± %.1fσ)", ci_z))
    graphics::lines(x_s, mu_s, col = "red", lwd = 2)
    graphics::polygon(c(x_s, rev(x_s)), c(upper, rev(lower)),
                      col = grDevices::rgb(0.2, 0.2, 1, alpha = 0.2), border = NA)

    # Add legend
    legend("topleft",
           legend = c("Data", "Mean (μ)", sprintf("%.0f%% CI", 100*(2*pnorm(ci_z)-1))),
           col = c("black", "red", grDevices::rgb(0.2, 0.2, 1, alpha = 0.4)),
           pch = c(16, NA, NA),
           lty = c(NA, 1, NA),
           lwd = c(NA, 2, NA),
           fill = c(NA, NA, grDevices::rgb(0.2, 0.2, 1, alpha = 0.2)),
           border = NA,
           bty = "n")

    return(invisible(object))
  }

  # ---- (3) Multiple Features: Partial Effect Plots ----
  if (isTRUE(feature_plots) && p > 1) {
    cat("\n(2) Partial Effects per Feature\n")
    cat("------------------------------\n")

    n_show <- min(p, max_features)
    if (n_show < p) {
      cat(sprintf("Showing first %d of %d features.\n", n_show, p))
    }

    # Setup plot grid
    n_cols <- min(3, n_show)
    n_rows <- ceiling(n_show / n_cols)
    graphics::par(mfrow = c(n_rows, n_cols), mar = c(4, 4, 3, 1))

    for (j in 1:n_show) {
      x_j <- X_used[, j]
      x_range <- seq(min(x_j), max(x_j), length.out = 100)

      # Get feature network outputs
      feat_out <- matrix(NA, nrow = 2, ncol = 100)
      for (i in 1:100) {
        x_single <- matrix(x_range[i], nrow = 1, ncol = 1)
        fwd_j <- forward_feature_net(x_single, object$params, j)
        feat_out[, i] <- fwd_j$output
      }

      # Add global biases
      mu_j <- feat_out[1, ] + object$params$beta_mu
      sigma_j <- Softplus_(feat_out[2, ] + object$params$beta_sigma)

      # Denormalize if target parameters provided
      if (!is.null(target_mean) && !is.null(target_sd)) {
        mu_j <- target_mean + target_sd * mu_j
        sigma_j <- target_sd * sigma_j
      }

      # Plot
      plot(x_range, mu_j, type = "l", col = "blue", lwd = 2,
           xlab = x_cols[j], ylab = "Partial Effect",
           main = paste("Feature:", x_cols[j]),
           ylim = range(c(mu_j - ci_z*sigma_j, mu_j + ci_z*sigma_j)))

      # Add uncertainty bands
      polygon(c(x_range, rev(x_range)),
              c(mu_j + ci_z*sigma_j, rev(mu_j - ci_z*sigma_j)),
              col = grDevices::rgb(0.2, 0.2, 1, alpha = 0.2), border = NA)

      # Add rug plot for data density
      rug(x_j, side = 1, col = grDevices::rgb(0, 0, 0, alpha = 0.3))
    }

    graphics::par(mfrow = c(1,1))  # Reset plot parameters
  }

  return(invisible(object))
}

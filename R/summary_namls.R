# ---- Helper -------------------------------------------------------------
Softplus_ <- function(z) log1p(exp(-abs(z))) + pmax(z, 0)

#' Summary of a Trained NAMLS Model
#'
#' Provides a summary of a trained NAMLS model, including its architecture,
#' training setup, training results, and visualizations of model predictions.
#' Depending on the number of input features, it produces loss curves,
#' 1D fits with confidence intervals, or partial effect plots for multiple features.
#'
#' @param object Trained NAMLS model object containing parameters, optimizer, learning rate,
#'   epochs, losses, and optional normalization info.
#' @param data Data frame used for model evaluation, including predictors and target.
#' @param target_col Character string with the name of the target variable.
#' @param show_plot Logical; if TRUE, plots of losses and predictions are shown. Default is TRUE.
#' @param yscale Character; determines y-axis scaling for loss plots:
#'   `"auto"` (linear), `"log"` (logarithmic), `"robust"` (capped at quantile).
#' @param cap_quantile Numeric (0–1); quantile for robust loss capping. Default is 0.99.
#' @param drop_first Integer; number of initial epochs to exclude from loss plots. Default is 0.
#' @param feature_plots Logical; if TRUE and more than one feature is present, partial effect
#'   plots are generated. Default is TRUE.
#' @param max_features Integer; maximum number of features to display in partial effect plots. Default is 6.
#' @param ci_z Numeric; z-value used for confidence interval calculation. Default is 1.96 (≈ 95% CI).
#'
#' @return Invisibly returns the input `object` after printing a summary and
#'   optionally plotting results.
#'
#' @examples
#' \dontrun{
#' # Assuming `namls_model` is a trained NAMLS model object
#' summary.NAMLS(
#'   object = namls_model,
#'   data = mydata,
#'   target_col = "y",
#'   show_plot = TRUE,
#'   yscale = "robust",
#'   feature_plots = TRUE
#' )
#' }
#'
#' @export
#' @method summary NAMLS

summary.NAMLS <- function(object,
                           data,
                           target_col,
                           show_plot = TRUE,
                           yscale = c("auto","log","robust"),
                           cap_quantile = 0.99,
                           drop_first = 0,
                           feature_plots = TRUE,
                           max_features = 6,
                           ci_z = 1.96) {
  yscale <- match.arg(yscale)

  # -- interne Plot-Funktion für Loss --
  plot_losses_ <- function(tr, vl, yscale, cap_q, drop_first) {
    tr <- as.numeric(tr); vl <- if (!is.null(vl)) as.numeric(vl) else NULL
    if (drop_first > 0 && length(tr) > drop_first) {
      idx <- (drop_first + 1):length(tr)
      tr <- tr[idx]; if (!is.null(vl)) vl <- vl[idx]
    }
    if (length(tr) == 0) return(invisible())

    epochs <- seq_along(tr)
    loss_all <- if (!is.null(vl)) c(tr, vl) else tr

    if (yscale == "log") {
      min_pos <- min(loss_all[loss_all > 0], na.rm = TRUE)
      tr2 <- pmax(tr, min_pos * 1e-6)
      vl2 <- if (!is.null(vl)) pmax(vl, min_pos * 1e-6) else NULL
      graphics::par(mfrow = c(1,1))
      graphics::plot(epochs, tr2, type="l", log="y",
                     main="Training vs. Validation Loss",
                     xlab="Epoch", ylab="Loss", col="blue")
      if (!is.null(vl2)) {
        graphics::lines(epochs, vl2, lty=2, col="red")
        graphics::legend("topright", c("Train","Validation"),
                         lty=c(1,2), col=c("blue","red"), bty="n")
      }
    } else if (yscale == "robust") {
      cap <- stats::quantile(loss_all, cap_q, na.rm = TRUE)
      tr2 <- pmin(tr, cap)
      vl2 <- if (!is.null(vl)) pmin(vl, cap) else NULL
      rng <- range(c(tr2, vl2), finite = TRUE)
      graphics::par(mfrow = c(1,1))
      graphics::plot(epochs, tr2, type="l", ylim=rng,
                     main="Training vs. Validation Loss",
                     xlab="Epoch", ylab="Loss", col="blue")
      if (!is.null(vl2)) {
        graphics::lines(epochs, vl2, lty=2, col="red")
        graphics::legend("topright", c("Train","Validation"),
                         lty=c(1,2), col=c("blue","red"), bty="n")
      }
    } else {
      rng <- range(loss_all, finite = TRUE)
      graphics::par(mfrow = c(1,1))
      graphics::plot(epochs, tr, type="l", ylim=rng,
                     main="Training vs. Validation Loss",
                     xlab="Epoch", ylab="Loss", col="blue")
      if (!is.null(vl)) {
        graphics::lines(epochs, vl, lty=2, col="red")
        graphics::legend("topright", c("Train","Validation"),
                         lty=c(1,2), col=c("blue","red"), bty="n")
      }
    }
  }

  graphics::par(mfrow = c(1,1), mar = c(5,4,2,1) + 0.1, oma = c(0,0,0,0))


  # -- Architektur-Print --
  arch_print_ <- function(object) {
    cat("Feature networks:       ", object$n_features, "\n", sep="")
    if (!is.null(object$architecture$layer_sizes)) {
      ls <- object$architecture$layer_sizes
      cat("Subnet architecture:     ", paste(ls, collapse=" -> "), "\n", sep="")
    } else if (!is.null(object$architecture$n_h)) {
      cat("Subnet architekture:     ",
          paste(c(1, object$architecture$n_h, 2), collapse=" -> "), "\n", sep="")
    }
    cat("Optimizer:              ", object$optimizer, "\n", sep="")
    cat("Loss function:           Negative Log-Likelihood\n")
    cat("Learning rate (start):  ", object$lr, "\n", sep="")
    if (!is.null(object$final_lr)) cat("Learning rate (final):   ", object$final_lr, "\n", sep="")
    cat("Trained epochs:         ", length(object$train_loss), "\n", sep="")
    cat(sprintf("Final training loss:     %.6f\n", tail(object$train_loss, 1)))
    if (!is.null(object$val_loss))      cat(sprintf("Final validation loss:   %.6f\n", tail(object$val_loss, 1)))
    if (!is.null(object$best_val_loss)) cat(sprintf("Best validation loss:    %.6f\n", object$best_val_loss))
  }

  # ---- Kopf ----
  cat("==============================\n")
  cat("-- NAMLS Model Summary --\n")
  cat("==============================\n\n")
  arch_print_(object)

  # ---- (1) Loss-Plot in RStudio-Plot-Pane ----
  if (isTRUE(show_plot)) {
    cat("\n(1) Loss-Curves\n------------------------------\n")
    plot_losses_(object$train_loss, object$val_loss, yscale, cap_quantile, drop_first)
  }

  # Keine weiteren Plots gewünscht
  if (!isTRUE(show_plot)) return(invisible(object))

  # ---- Daten vorbereiten ----
  x_cols <- setdiff(names(data), target_col)
  if (length(x_cols) == 0) {
    warning("No feature-columns found (data without predictors?).")
    return(invisible(object))
  }
  X_raw <- as.matrix(data[, x_cols, drop = FALSE])
  if (!is.null(object$normalization)) {
    X_used <- scale(X_raw, center = object$normalization$mean, scale = object$normalization$sd)
  } else {
    X_used <- X_raw
  }
  X_t <- t(X_used)
  p <- ncol(X_raw)

  if (!is.null(object$n_features) && p != object$n_features) {
    stop(sprintf("Number of features in 'data' (%d) ≠ object$n_features (%d). Check order/columns!",
                 p, object$n_features))
  }

  # ---- Vollvorhersage ----
  fwd <- forward_namls(X_t, object$params, dropout_rate = 0, training = FALSE)
  mu_full    <- as.numeric(fwd$mu)
  sigma_full <- as.numeric(fwd$sigma)

  # ---- (2) 1-Feature-Plot (Fit + CI) im gleichen Pane ----
  if (p == 1) {
    cat("\n(2) Prediction plot (1 Feature)\n------------------------------\n")
    y <- data[[target_col]]
    x <- X_raw[, 1]
    ord <- order(x)
    x_s <- x[ord]; y_s <- y[ord]
    mu_s <- mu_full[ord]; sig_s <- sigma_full[ord]
    upper <- mu_s + ci_z * sig_s
    lower <- mu_s - ci_z * sig_s

    graphics::par(mfrow = c(1,1))
    graphics::plot(x_s, y_s, pch=16, cex=0.6,
                   xlab = x_cols[1], ylab = target_col,
                   main = "NAMLS Fit (μ with 95% CI)")
    graphics::lines(x_s, mu_s, col="red", lwd=2)
    graphics::polygon(c(x_s, rev(x_s)), c(upper, rev(lower)),
                      col = grDevices::rgb(0.2, 0.2, 1, alpha=0.2), border=NA)
    return(invisible(object))
  }

  # ---- (2) Partielle Effekte (alle in EINEM Multi-Panel-Plot im Pane) ----
  if (isTRUE(feature_plots) && p > 1) {
    cat("\n(2) Partial effects per feature\n------------------------------\n")
    if (!exists("forward_feature_net", mode="function")) {
      warning("forward_feature_net() not found – Feature plots are skipped.")
      return(invisible(object))
    }

    # Beiträge berechnen
    mu_contrib_list     <- vector("list", p)
    sigraw_contrib_list <- vector("list", p)
    for (j in seq_len(p)) {
      xj <- matrix(X_t[j, ], nrow = 1)
      resj <- forward_feature_net(xj, object$params, j, dropout_rate = 0, training = FALSE)
      mu_contrib_list[[j]]     <- as.numeric(resj$output[1, ])
      sigraw_contrib_list[[j]] <- as.numeric(resj$output[2, ])
    }

    sigraw_mat   <- do.call(cbind, sigraw_contrib_list)  # N x p
    sigraw_means <- colMeans(sigraw_mat)
    eps <- if (!is.null(object$params$sigma_floor)) object$params$sigma_floor else 0

    k <- min(p, max_features)
    nrow_plot <- ceiling(k / 2)
    graphics::par(
      mfrow = c(nrow_plot, 2),
      mar   = c(3, 3, 1.5, 2.8),  # bottom, left, top, right (kleiner als vorher)
      oma   = c(0, 0, 0, 0),
      mgp   = c(1.7, 0.5, 0),     # Achsentitel/-ticks näher an die Achse
      tcl   = -0.2,               # kürzere Tickmarks
      cex   = 0.85                # leicht kleinere Schrift
    )


    for (j in seq_len(k)) {
      xj  <- X_raw[, j]; ord <- order(xj)
      xjs <- xj[ord]
      mu_j <- mu_contrib_list[[j]][ord]

      base_sigraw <- object$params$beta_sigma + sum(sigraw_means) - sigraw_means[j]
      sig_partial <- Softplus_( base_sigraw + sigraw_contrib_list[[j]][ord] ) + eps

      graphics::plot(xjs, mu_j, type="l", lwd=2, col="black",
                     xlab = x_cols[j], ylab = expression(f[mu](x[j])),
                     main = paste0("Feature ", j, ": μ-contribution & σ(partial)"))
      graphics::grid()
      graphics::par(new = TRUE)
      graphics::plot(xjs, sig_partial, type="l", lwd=1.5, lty=2, col="red",
                     axes=FALSE, xlab="", ylab="")
      graphics::axis(4); graphics::mtext("sigma (partial)", side=4, line=2)
      graphics::legend("topleft", c("μ-contribution","σ (partial)"),
                       lty=c(1,2), lwd=c(2,1.5), col=c("black","red"), bty="n")
    }
  }

  invisible(object)
}


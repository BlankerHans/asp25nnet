#' Summary für NAMLSS-Objekte (NAM mit Location & Scale)
#'
#' Zeigt Trainings-Setup, Loss-Kurven (Train/Val) und – falls \code{show_plot=TRUE} –
#' eine Vorhersage-Visualisierung (bei 1 Feature) bzw. partielle Effekte je Feature
#' (bei >1 Features): \eqn{g_{\mu,j}(x_j)} und eine partielle \eqn{\sigma(x_j)}-Kurve,
#' wobei die übrigen Feature-Beiträge für \eqn{\sigma} auf ihrem Datensatz-Mittel fixiert werden.
#'
#' @param object Ein \code{NAMLSS}-Objekt (siehe \code{train_namlss}).
#' @param data   Datensatz (data.frame) mit Features und Zielspalte.
#' @param target_col Name der Zielspalte (Character).
#' @param show_plot Ob Plots erstellt werden sollen (Default: TRUE).
#' @param yscale Skalierung der Loss-Kurven: "auto", "log" oder "robust".
#' @param cap_quantile Quantil-Cutoff für "robust" (Default: 0.99).
#' @param drop_first Anzahl an Epochen, die am Anfang aus den Loss-Plots entfernt werden.
#' @param feature_plots Ob bei >1 Features partielle Effekte geplottet werden (Default: TRUE).
#' @param max_features Max. Anzahl Features, die als partielle Effekte gezeigt werden (Default: 6).
#' @param ci_z z-Wert für das Konfidenzintervall (z. B. 1.96 für 95%).
#'
#' @export
#' @method summary NAMLSS
summary.NAMLSS <- function(object,
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

  # --- Hilfsfunktionen (lokal, kollidieren nicht mit globalen) ---
  Softplus_ <- function(z) log1p(exp(-abs(z))) + pmax(z, 0)

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
      graphics::plot(epochs, tr2, type = "l", log = "y",
                     main = "Training vs. Validation Loss",
                     xlab = "Epoch", ylab = "Loss", col = "blue")
      if (!is.null(vl2)) {
        graphics::lines(epochs, vl2, lty = 2, col = "red")
        graphics::legend("topright", c("Train","Validation"),
                         lty = c(1,2), col = c("blue","red"), bty = "n")
      }
    } else if (yscale == "robust") {
      cap <- stats::quantile(loss_all, cap_q, na.rm = TRUE)
      tr2 <- pmin(tr, cap)
      vl2 <- if (!is.null(vl)) pmin(vl, cap) else NULL
      rng <- range(c(tr2, vl2), finite = TRUE)
      graphics::plot(epochs, tr2, type = "l", ylim = rng,
                     main = "Training vs. Validation Loss",
                     xlab = "Epoch", ylab = "Loss", col = "blue")
      if (!is.null(vl2)) {
        graphics::lines(epochs, vl2, lty = 2, col = "red")
        graphics::legend("topright", c("Train","Validation"),
                         lty = c(1,2), col = c("blue","red"), bty = "n")
      }
    } else {
      rng <- range(loss_all, finite = TRUE)
      graphics::plot(epochs, tr, type = "l", ylim = rng,
                     main = "Training vs. Validation Loss",
                     xlab = "Epoch", ylab = "Loss", col = "blue")
      if (!is.null(vl)) {
        graphics::lines(epochs, vl, lty = 2, col = "red")
        graphics::legend("topright", c("Train","Validation"),
                         lty = c(1,2), col = c("blue","red"), bty = "n")
      }
    }
  }

  arch_print_ <- function(object) {
    cat("Feature-Networks:       ", object$n_features, "\n", sep = "")
    if (!is.null(object$architecture$layer_sizes)) {
      ls <- object$architecture$layer_sizes
      cat("Subnet-Architektur:     ", paste(ls, collapse = " -> "), "\n", sep = "")
    } else if (!is.null(object$architecture$n_h)) {
      cat("Subnet-Architektur:      ", paste(c(1, object$architecture$n_h, 2), collapse = " -> "), "\n", sep = "")
    }
    cat("Optimizer:              ", object$optimizer, "\n", sep = "")
    cat("Loss function:           Negative Log-Likelihood\n")
    cat("Learning rate (start):  ", object$lr, "\n", sep = "")
    if (!is.null(object$final_lr)) cat("Learning rate (final):   ", object$final_lr, "\n", sep = "")
    cat("Trained epochs:         ", length(object$train_loss), "\n", sep = "")
    cat(sprintf("Final training loss:     %.6f\n", tail(object$train_loss, 1)))
    if (!is.null(object$val_loss)) cat(sprintf("Final validation loss:   %.6f\n", tail(object$val_loss, 1)))
    if (!is.null(object$best_val_loss)) cat(sprintf("Best validation loss:    %.6f\n", object$best_val_loss))
  }

  # --- Summary Kopf ---
  cat("-- NAMLSS Model Summary --\n")
  cat("==============================\n\n")
  arch_print_ (object)

  # --- Loss Plot ---
  if (isTRUE(show_plot)) {
    cat("\n(1) Loss-Kurven\n------------------------------\n")
    old_par <- graphics::par(no.readonly = TRUE)
    on.exit(graphics::par(old_par), add = TRUE)
    graphics::par(mfrow = c(1,1))
    plot_losses_(object$train_loss, object$val_loss, yscale, cap_quantile, drop_first)
  }

  # --- Vorhersage-/Partielle Effekte ---
  if (!isTRUE(show_plot)) return(invisible(object))

  x_cols <- setdiff(names(data), target_col)
  X_raw <- as.matrix(data[, x_cols, drop = FALSE])

  # Optionale Normalisierung (falls im Objekt enthalten)
  if (!is.null(object$normalization)) {
    X_used <- scale(X_raw, center = object$normalization$mean, scale = object$normalization$sd)
  } else {
    X_used <- X_raw
  }

  X_t <- t(X_used)  # (features x N)
  N <- nrow(X_raw); p <- ncol(X_raw)

  # Konsistenz-Check
  if (!is.null(object$n_features) && p != object$n_features) {
    warning(sprintf("Anzahl Features im data (%d) ungleich object$n_features (%d). Prüfe die Reihenfolge/Spalten!", p, object$n_features))
  }

  # Vollvorhersage
  fwd <- forward_namlss(X_t, object$params, dropout_rate = 0, training = FALSE)
  mu_full <- as.numeric(fwd$mu)
  sigma_full <- as.numeric(fwd$sigma)

  # 1 Feature: Scatter + Fit + CI
  if (p == 1) {
    cat("\n(2) Vorhersage-Plot (1 Feature)\n------------------------------\n")
    y <- data[[target_col]]
    x <- X_raw[,1]
    ord <- order(x)
    x_s <- x[ord]; y_s <- y[ord]
    mu_s <- mu_full[ord]
    sig_s <- sigma_full[ord]
    upper <- mu_s + ci_z * sig_s
    lower <- mu_s - ci_z * sig_s

    graphics::plot(x_s, y_s, pch = 16, cex = 0.6,
                   xlab = x_cols[1], ylab = target_col,
                   main = "NAMLSS Fit (μ mit 95% CI)")
    graphics::lines(x_s, mu_s, col = "red", lwd = 2)
    graphics::polygon(c(x_s, rev(x_s)), c(upper, rev(lower)),
                      col = grDevices::rgb(0.2, 0.2, 1, alpha = 0.2), border = NA)
    invisible(object); return(object)
  }

  # >1 Features: Partielle Effekte
  if (isTRUE(feature_plots)) {
    cat("\n(2) Partielle Effekte je Feature\n------------------------------\n")

    # Beiträge je Feature vorab berechnen (effizient)
    mu_contrib_list <- vector("list", p)
    sigraw_contrib_list <- vector("list", p)
    for (j in seq_len(p)) {
      xj <- matrix(X_t[j, ], nrow = 1)
      resj <- forward_feature_net(xj, object$params, j, dropout_rate = 0, training = FALSE)
      mu_contrib_list[[j]] <- as.numeric(resj$output[1, ])
      sigraw_contrib_list[[j]] <- as.numeric(resj$output[2, ])
    }

    # Basis für sigma: andere Beiträge im Mittel fixieren
    sigraw_mat <- do.call(cbind, sigraw_contrib_list)  # (N x p)
    # Mittel über N für jeden Feature-Beitrag (Vektor p)
    sigraw_means <- colMeans(sigraw_mat)

    old_par2 <- graphics::par(no.readonly = TRUE)
    on.exit(graphics::par(old_par2), add = TRUE)

    k <- min(p, max_features)
    nrow_plot <- ceiling(k / 2)
    graphics::par(mfrow = c(nrow_plot, 2), mar = c(4,4,2,4))

    for (j in seq_len(k)) {
      xj <- X_raw[, j]
      ord <- order(xj)
      xjs <- xj[ord]
      mu_j <- mu_contrib_list[[j]][ord]

      # sigma-partial: beta_sigma + mean(other sigraw) + sigraw_j(xj)
      base_sigraw <- object$params$beta_sigma + sum(sigraw_means) - sigraw_means[j]
      sig_partial <- Softplus_( base_sigraw + sigraw_contrib_list[[j]][ord] )

      rngL <- range(mu_j, finite = TRUE)
      graphics::plot(xjs, mu_j, type = "l", lwd = 2, col = "black",
                     xlab = x_cols[j], ylab = expression(g[mu](x[j])),
                     main = paste0("Feature ", j, ": μ-Beitrag & σ(partial)"))
      graphics::grid()

      graphics::par(new = TRUE)
      graphics::plot(xjs, sig_partial, type = "l", lwd = 1.5, lty = 2, col = "red",
                     axes = FALSE, xlab = "", ylab = "")
      graphics::axis(4)
      graphics::mtext("sigma (partial)", side = 4, line = 2)
      graphics::legend("topleft", c("μ-Beitrag", "σ (partial)"),
                       lty = c(1,2), lwd = c(2,1.5), col = c("black","red"), bty = "n")
    }
  }

  invisible(object)
}

#' Summary für NN-Objekte
#' @param object Ein NN-Objekt
#' @param ... weitere Argumente
#' @export
#' @method summary NN
summary.NN <- function(object,
                       show_plot = TRUE,
                       yscale = c("auto","log","robust"),
                       cap_quantile = 0.99,
                       drop_first = 0
                       ) {
  yscale <- match.arg(yscale)

  cat("--Model Summary--\n")
  cat("==============================\n\n")
  cat("Training Setup \n")
  cat("------------------------------\n")
  cat("\tOptimizer:              ", object$optimizer, "\n")
  cat("\tLoss function:          ", "Negative Log-Likelihood\n")
  cat("\tLearning rate:          ", object$lr, "\n")
  cat("\tNumber of epochs:       ", object$epochs, "\n\n")

  ## Architektur-Details
  plot_architecture(object)

  cat("\n")
  cat("Training Results \n")
  cat("------------------------------\n")
  cat("\tTrained Epochs:         ", length(object$train_loss), "\n")
  cat(sprintf("\tFinal training loss:     %.3f\n", tail(object$train_loss, 1)))
  if (!is.null(object$val_loss)) {
    cat(sprintf("\tFinal validation loss:   %.3f\n", tail(object$val_loss, 1)))
  }

  #Preperation of loss vectors
  tl <- as.numeric(object$train_loss)
  vl <- if (!is.null(object$val_loss)) as.numeric(object$val_loss) else NULL

  #If requested, cut of first x epochs
  if (drop_first > 0 && length(tl) > drop_first) {
    idx <- (drop_first + 1):length(tl); tl <- tl[idx]; if (!is.null(vl)) vl <- vl[idx]
  }

  #If requested (and loss-data exists), create plot
  if (isTRUE(show_plot) && length(tl) > 0) {
    epochs   <- seq_along(tl)
    loss_all <- if (!is.null(vl)) c(tl, vl) else tl

    # Case 1 ("log"): plots losses on a log scale
    if (yscale == "log") {
      min_pos <- min(loss_all[loss_all > 0], na.rm = TRUE)
      tl2 <- pmax(tl, min_pos * 1e-6)
      vl2 <- if (!is.null(vl)) pmax(vl, min_pos * 1e-6) else NULL

      graphics::plot(epochs, tl2, type = "l", log = "y",
                     main = "Training vs. Validation Loss",
                     xlab = "Epoch", ylab = "Loss",
                     col = "blue")                          # <-- Farbe
      if (!is.null(vl2)) {
        graphics::lines(epochs, vl2, lty = 2, col = "red")  # <-- Farbe
        graphics::legend("topright", c("Train","Validation"),
                         lty = c(1,2), col = c("blue","red"), bty = "n")
      }

      # Case 2 ("robust"): losses are capped at a chosen quantile
    } else if (yscale == "robust") {
      cap <- stats::quantile(loss_all, cap_quantile, na.rm = TRUE)
      tl2 <- pmin(tl, cap)
      vl2 <- if (!is.null(vl)) pmin(vl, cap) else NULL
      rng <- range(c(tl2, vl2), finite = TRUE)

      graphics::plot(epochs, tl2, type = "l", ylim = rng,
                     main = "Training vs. Validation Loss",
                     xlab = "Epoch", ylab = "Loss",
                     col = "blue")                          # <-- Farbe
      if (!is.null(vl2)) {
        graphics::lines(epochs, vl2, lty = 2, col = "red")  # <-- Farbe
        graphics::legend("topright", c("Train","Validation"),
                         lty = c(1,2), col = c("blue","red"), bty = "n")
      }

      # Case 3 ("auto"): uses the standard linear scale without adjustments
    } else { # auto
      rng <- range(loss_all, finite = TRUE)
      graphics::plot(epochs, tl, type = "l", ylim = rng,
                     main = "Training vs. Validation Loss",
                     xlab = "Epoch", ylab = "Loss",
                     col = "blue")                          # <-- Farbe
      if (!is.null(vl)) {
        graphics::lines(epochs, vl, lty = 2, col = "red")   # <-- Farbe
        graphics::legend("topright", c("Train","Validation"),
                         lty = c(1,2), col = c("blue","red"), bty = "n")
      }
    }



  }

  invisible(object)
}

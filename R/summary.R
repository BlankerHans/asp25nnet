summary <- function(model, plot = TRUE) {

  # Summary Plots

  if (plot) {
    epochs <- seq_along(model$train_loss)

    if (!is.null(model$val_loss)) {
      rng <- range(c(model$train_loss, model$val_loss))
    } else {
      rng <- range(model$train_loss)
    }

    # Summary Plot Training
    plot(
      epochs, model$train_loss, type = "l",
      col  = "blue",
      ylim = rng,
      main = "Training vs. Validation Loss",
      xlab = "Epoch",
      ylab = "Loss"
    )

    if (!is.null(model$val_loss)) {
      lines(epochs, model$val_loss, col = "red", lty = 2)
      legend(
        "topright",
        legend = c("Train", "Validation"),
        col    = c("blue", "red"),
        lty    = c(1, 2),
        bty    = "n"
      )
    }
  }

  invisible(NULL)
}

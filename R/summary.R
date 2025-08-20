summary.NN <- function(object, plot = TRUE) {

  cat("--Model Summary-- \n")
  cat("==============================\n\n")
  # Summary layer architecture (activation, Nr of params per layer)
  # Summary Training setup (optimizer, batch size, epochs, learning rate)
  cat("Training Setup \n")
  cat("------------------------------\n")
  cat("\tOptimizer:              ", object$optimizer, "\n")
  cat("\tLoss function:          ", "Negative Log-Likelihood\n") #<-generalisieren falls Wahlmöglichkeit implememtiert!!
  cat("\tLearning rate:          ", object$lr, "\n")
  #cat("batch size:            ", objects$batch_size, "\n")
  cat("\tNumber of epochs:       ", object$epochs, "\n\n")

  # Summary Training (final train, val & test loss)
  cat("Training Results \n")
  cat("------------------------------\n")
  cat("\tTrained Epochs:         ", length(object$train_loss) , "\n") #Für early stopping
  cat(sprintf("\tFinal training loss:     %.3f\n", tail(object$train_loss, 1)))
  if (!is.null(object$val_loss)) {
  cat(sprintf("\tFinal validation loss:   %.3f\n", tail(object$val_loss, 1)))

    }
  #cat("Loss on test set:       ", , "\n")

  # Summary Plots (anpassen dass 3D graph bei 2 Inputs und NAM Vorschlag bei >2 Inputs)

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

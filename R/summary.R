#' Summary of a Trained Neural Network Model
#'
#' Provides a summary of a trained neural network, including its
#' architecture, training setup, losses, and visualizations of model fit.
#' Depending on the number of input features, it generates either 2D or 3D plots.
#'
#' @param object Trained neural network model object containing parameters,
#'   optimizer, learning rate, epochs, losses, and optional normalization info.
#' @param data Data frame used for model evaluation, including predictors and target.
#' @param target_col Character string with the name of the target variable.
#' @param show_plot Logical; if TRUE, plots of losses and predictions are shown. Default is TRUE.
#' @param yscale Character; determines y-axis scaling for loss plots:
#'   `"auto"` (linear), `"log"` (logarithmic), `"robust"` (capped at quantile).
#' @param cap_quantile Numeric (0–1); quantile for robust loss capping. Default is 0.99.
#' @param drop_first Integer; number of initial epochs to exclude from loss plots. Default is 0.
#'
#' @return Invisibly returns the input `object` after printing and optionally plotting
#'   summaries, losses, and model predictions.
#'
#' @examples
#' \dontrun{
#' # Assuming `nn_model` is a trained neural network object
#' summary.NN(
#'   object = nn_model,
#'   data = mydata,
#'   target_col = "y",
#'   show_plot = TRUE,
#'   yscale = "auto"
#' )
#' }
#'
#' @export


summary.NN <- function(object,
                       data,
                       target_col,
                       show_plot = TRUE,
                       yscale = c("auto","log","robust"),
                       cap_quantile = 0.99,
                       drop_first = 0
) {
  yscale <- match.arg(yscale)

  cat("--Model Summary--\n")
  cat("==============================\n\n")

  ## NN architecture
  plot_architecture(object)

  cat("\n\nTraining Setup \n")
  cat("------------------------------\n")
  cat("\tOptimizer:              ", object$optimizer, "\n")
  cat("\tLoss function:          ", "Negative Log-Likelihood\n")
  cat("\tLearning rate:          ", object$lr, "\n")
  cat("\tNumber of epochs:       ", object$epochs, "\n\n")

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


    # == Data Preparation for prediction plot ==
    x_col <- setdiff(names(data), target_col)
    X <- data[, x_col, drop = FALSE]

    # If needed, normalize features based on mean, sd of training set
    if (!is.null(object$normalization)) {
      X <- scale(X, center = object$normalization$mean, scale = object$normalization$sd)
    }

    X <- t(as.matrix(X)) # transpose matrix for forward pass
    nr_inputs <- length(x_col)

    # Forward pass with trained parameters
    fwd <- forward(X, object$params)

    #mu and sigma according to forward pass
    mu <- as.numeric(fwd$mu)
    sigma <- exp(as.numeric(fwd$log_sigma))

    #boundaries for CI
    upper <- mu + 1.96 * sigma
    lower <- mu - 1.96 * sigma

    # Prediction plot: 1 Input
    if (nr_inputs == 1) {
      #Sort by input
      ord <- order(data[[x_col]])
      x_sorted <- data[[x_col]][ord]
      y_sorted <- data[[target_col]][ord]
      mu_sorted <- mu[ord]
      upper_sorted <- upper[ord]
      lower_sorted <- lower[ord]

      #Plot
      plot(x_sorted, y_sorted,
           xlab = x_col, ylab = target_col,
           main = paste("Model fit on", target_col, "vs", x_col))

      #line for mu
      lines(x_sorted, mu_sorted, col = "red", lwd = 2)

      #add CI
      polygon(c(x_sorted, rev(x_sorted)),
              c(upper_sorted, rev(lower_sorted)),
              col = rgb(0.2, 0.2, 1, alpha = 0.2),
              border = NA)
    }


    #Prediction plot: 2 Inputs
    else if (nr_inputs == 2) {

      x1 <- data[[x_col[1]]]
      x2 <- data[[x_col[2]]]
      y  <- data[[target_col]]

      # Grid für Vorhersagen
      grid_size <- 40
      x1_seq <- seq(min(x1), max(x1), length.out = grid_size)
      x2_seq <- seq(min(x2), max(x2), length.out = grid_size)

      grid <- expand.grid(x1_seq, x2_seq)
      names(grid) <- x_col

      # Normalisieren falls nötig
      if (!is.null(object$normalization)) {
        grid <- scale(grid,
                      center = object$normalization$mean[x_col],
                      scale  = object$normalization$sd[x_col])
      }

      X_grid <- t(as.matrix(grid))

      # Forward pass
      fwd_grid   <- forward(X_grid, object$params)
      mu_grid    <- matrix(as.numeric(fwd_grid$mu), nrow = grid_size, byrow = FALSE)

      # Plot
      rgl::open3d()
      rgl::points3d(x1, x2, y, col = "black", size = 5)  # Datenpunkte

      # Fläche: mu
      rgl::surface3d(x1_seq, x2_seq, mu_grid, color = "red", alpha = 0.6, front = "lines")

      # Achsen + Titel
      rgl::axes3d()
      rgl::title3d(xlab = x_col[1],
              ylab = x_col[2],
              zlab = target_col,
              main = "3D plot with estimated mu ")

      cat("\nTo access the plot, use the command rgl::rglwidget()")
    }
    else {
      message("Plotting not possible for > 2 input features")
    }



  }
  invisible(object)
}



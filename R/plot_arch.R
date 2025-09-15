#' Plots architecture for DNN (NN class)
#'
#' Helper function to plot the model architecture after training
#'
#' @param model trained DNN model
#' @return network architecture
#'
#' @export
plot_architecture <- function(model) {
  if ("architecture" %in% names(model)) {
    arch <- model$architecture
  } else if ("n_h" %in% names(model)) {
    arch <- model
  } else {
    stop("Model must have architecture information")
  }

  # Create simple text visualization
  layers <- c(arch$n_x, arch$n_h, arch$n_y)
  layer_names <- c("Input",
                   paste0("Hidden ", seq_along(arch$n_h)),
                   "Output")

  cat("\nNetwork Architecture:\n")
  cat("====================\n")
  for (i in seq_along(layers)) {
    cat(sprintf("%-10s: %3d neurons\n", layer_names[i], layers[i]))
    if (i < length(layers)) {
      cat(sprintf("   \u2193  [%d \u00D7 %d weights]\n", layers[i+1], layers[i]))
    }
  }
  cat("\nTotal parameters:", sum(layers[-1] * layers[-length(layers)] + layers[-1]), "\n")
}

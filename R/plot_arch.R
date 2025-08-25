#' Plots architecture for DNN (NN class)
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
      cat(sprintf("    ↓  [%d × %d weights]\n", layers[i+1], layers[i]))
    }
  }
  cat("\nTotal parameters:", sum(layers[-1] * layers[-length(layers)] + layers[-1]), "\n")
}

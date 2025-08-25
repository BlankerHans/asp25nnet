# Eval function
eval.NN <- function(object, split_output) {

  params <- object$params

  #Obtain respective targets for features in test set
  test_df <- split_output$test
  df_targets <- object$targets #muss noch in die Liste der trainingsfunktionen!
  test_df_targets <- df_targets[as.integer(rownames(test_df))]

  # If needed, normalize features based on mean, sd of training set
  if (!is.null(object$normalization)) {
    Test_set <- scale(test_df, center = object$normalization$mean, scale = object$normalization$sd)
  }
  else {
    Test_set <- test_df
  }

  Test_set <- t(as.matrix(Test_set)) # transpose matrix for forward pass

  # Forward pass with trained params on test set
  fwd <- forward(Test_set, params)


  # Calculate loss
  loss <- neg_log_lik(
    test_df_targets, as.numeric(fwd$mu), as.numeric(fwd$log_sigma),
    reduction = "mean")

  # Calculate key figures
  rmse <- sqrt(mean((test_df_targets - fwd$mu)^2))
  mae  <- mean(abs(test_df_targets - fwd$mu))

  cat("Test Loss (NLL):", loss, "\n")
  cat("Test RMSE:", rmse, "\n")
  cat("Test MAE:", mae, "\n")

  # True vs Predicted Plot
#  plot(
#    test_df_targets, as.numeric(fwd$mu),
#    xlab = "True", ylab = "Predicted (mu)",
#    main = "True vs Predicted",
#    pch = 19, col = rgb(0, 0, 1, 0.5)
#  )
#  abline(0, 1, col = "red", lwd = 2, lty = 2)  # perfekte Vorhersage
}

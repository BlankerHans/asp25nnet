# Eval function
eval.NN <- function(object, split_output, verbose = TRUE) {

  params <- object$params

  #Obtain respective targets for features in test set
  Test_set <- split_output$test
  df_targets <- object$targets
  test_df_targets <- df_targets[as.integer(rownames(Test_set))]

  Test_set <- t(as.matrix(Test_set)) # transpose matrix for forward pass

  # Forward pass with trained params on test set
  fwd <- forward(Test_set, params)

  #mu and sigma according to forward pass
  mu <- as.numeric(fwd$mu)
  sigma <- exp(as.numeric(fwd$log_sigma))

  cover <- mean( (test_df_targets >= mu - 1.96*sigma) & (test_df_targets <= mu + 1.96*sigma) )

  # Calculate loss
  loss <- neg_log_lik(
    test_df_targets, as.numeric(mu), as.numeric(fwd$log_sigma),
    reduction = "mean")

  # Calculate key figures
  rmse <- sqrt(mean((test_df_targets - mu)^2))
  mae  <- mean(abs(test_df_targets - mu))

  #Konsolenoutput
  if (verbose) {
  cat("Test Loss (NLL):", loss, "\n")
  cat("Test RMSE:", rmse, "\n")
  cat("Test MAE:", mae, "\n")
  cat(sprintf("95%% coverage: %.3f\n", cover))
  }

  return(list(
    fwd = fwd,
    loss = loss,
    mu = mu,
    sigma = sigma,
    cover = cover,
    rmse = rmse,
    mae = mae,
    test_df_targets = test_df_targets
  ))
  # True vs Predicted Plot
#  plot(
#    test_df_targets, as.numeric(fwd$mu),
#    xlab = "True", ylab = "Predicted (mu)",
#    main = "True vs Predicted",
#    pch = 19, col = rgb(0, 0, 1, 0.5)
#  )
#  abline(0, 1, col = "red", lwd = 2, lty = 2)  # perfekte Vorhersage
}

#' Evaluate a Trained Neural Network Model
#'
#' Evaluates a trained neural network on a given test set. Performs a forward
#' pass, calculates prediction uncertainty, loss, and key performance metrics.
#' Optionally prints evaluation results to the console.
#'
#' @param object Trained model object containing learned parameters (`params`)
#'   and target values (`targets`).
#' @param split_output List containing train/test split data. The element
#'   `split_output$test` must contain the test set.
#' @param verbose Logical; if `TRUE` (default), prints evaluation metrics to
#'   the console.
#'
#' @return A list with the following elements:
#'  \item{fwd}{Forward pass results on the test set.}
#'  \item{loss}{Negative log-likelihood loss on the test set.}
#'  \item{mu}{Predicted means for the test samples.}
#'  \item{sigma}{Predicted standard deviations for the test samples.}
#'  \item{cover}{Proportion of true values within the 95 percent prediction interval.}
#'  \item{rmse}{Root mean squared error on the test set.}
#'  \item{mae}{Mean absolute error on the test set.}
#'  \item{test_df_targets}{True target values for the test set.}
#'
#' @examples
#' \dontrun{
#'   evaluation <- eval.DNN(model, split_output, verbose = TRUE)
#'}
#'
#' @export

eval.DNN <- function(object, split_output, verbose = TRUE) {

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
  rmse <- sqrt(mean((mu - test_df_targets)^2)) #checken ob das so richtig ist!!
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
}

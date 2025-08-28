#' Negative Log-Likelihood for a Normal Distribution
#'
#' Computes the negative log-likelihood of observations \code{y} assuming a normal
#' distribution with mean \code{mu} and log standard deviation \code{log_sigma}.
#'
#' @param y Numeric vector of observed values.
#' @param mu Numeric vector of expected means.
#' @param log_sigma Logarithm of the standard deviation.
#' @param reduction Character string specifying the reduction method:
#'   \code{"sum"} (default) returns the sum of individual losses,
#'   \code{"mean"} returns their mean,
#'   \code{"raw"} returns the vector of individual loss values.
#'
#' @return A numeric scalar or vector of negative log-likelihood values,
#'   depending on the \code{reduction} parameter.
#'
#' @details
#' For each observation \( y_i \), the function calculates the term:
#' \[
#' \log(\sigma) + \frac{(y_i - \mu_i)^2}{2 \sigma^2}
#' \]
#' where \(\sigma = \exp(\text{log_sigma})\).
#' The result is then aggregated based on the \code{reduction} argument.
#'
#' @examples
#' y <- c(1.5, 2.0, 1.8)
#' mu <- c(1.4, 2.1, 1.7)
#' log_sigma <- log(0.1)
#' neg_log_lik(y, mu, log_sigma, reduction = "mean")
#'
#' @export
neg_log_lik <- function(y, mu, log_sigma, reduction = c("sum","mean","raw")) {

  reduction <- match.arg(reduction)

  sigma <- exp(log_sigma)

  loss_i <- 0.5 * log(2*pi) + log_sigma + (y - mu)^2 / (2 * sigma^2)

  if (reduction == "sum") {
    return(sum(loss_i))
  } else if (reduction == "mean") {
    return(mean(loss_i))
  } else {  # reduction == "raw"
    return(loss_i)
  }
}

neg_log_lik_invgamma <- function(y, log_alpha, log_beta, reduction = c("sum","mean","raw")) {

  reduction <- match.arg(reduction)

  alpha <- exp(log_alpha)
  beta <- exp(log_beta)

  n <- length(y)

  # log(L(α, β|y)) = -n(α + 1)log(y) - n*log(Γ(α)) + n*α*log(β) - Σ(β/y_i)
  # Für neg log-lik multiplizieren wir mit -1

  loss_i <- (alpha + 1) * log(y) + lgamma(alpha) - alpha * log(beta) + beta/y

  if (reduction == "sum") {
    return(sum(loss_i))
  } else if (reduction == "mean") {
    return(mean(loss_i))
  } else {  # reduction == "raw"
    return(loss_i)
  }
}

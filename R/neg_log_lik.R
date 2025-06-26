neg_log_lik <- function(y, mu, log_sigma, reduction = c("sum","mean","raw")) {

  reduction <- match.arg(reduction)

  sigma <- exp(log_sigma)

  loss_i <- log_sigma + (y - mu)^2 / (2 * sigma^2)

  if (reduction == "sum") {
    return(sum(loss_i))
  } else if (reduction == "mean") {
    return(mean(loss_i))
  } else {  # reduction == "raw"
    return(loss_i)
  }
}

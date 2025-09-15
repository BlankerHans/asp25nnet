#' Predict method for NAMLS objects
#'
#' @param object A NAMLS object.
#' @param newdata Optional data frame with new observations.
#' @param type Character, one of \code{c("link","response")}.
#' @param return_contributions Logical; if \code{TRUE}, also return contributions.
#' @param ... Further arguments (ignored), kept for generic consistency.
#' @return A numeric vector or data.frame with predictions.
#' @export
#' @importFrom stats predict
predict_namls <- function(object, newdata, type = c("response", "link"),
                          return_contributions = FALSE, ...) {
  type <- match.arg(type)

  # 1) Feature-Preprocessing wie im Training
  x_cols <- colnames(newdata)
  X_raw  <- as.data.frame(newdata)
  if (!is.null(object$preprocessing) &&
      identical(object$preprocessing$x_scaler_type, "pm1")) {
    X_used_df <- transform_pm1(
      X_raw, object$preprocessing$x_scaler,
      clip = isTRUE(object$preprocessing$clip)
    )
  } else if (!is.null(object$normalization)) {
    X_used_df <- as.data.frame(scale(as.matrix(X_raw),
                                     center = object$normalization$mean,
                                     scale  = object$normalization$sd))
  } else {
    X_used_df <- X_raw
  }
  X_t <- t(as.matrix(X_used_df))

  # 2) Forward
  fwd <- forward_namls(X_t, object$params, dropout_rate = 0, training = FALSE)
  mu_z <- as.numeric(fwd$mu)
  sig_z <- if (!is.null(fwd$sigma)) as.numeric(fwd$sigma) else exp(as.numeric(fwd$log_sigma))

  # 3) Skalenwahl
  if (type == "link") {
    out <- list(mu = mu_z, sigma = sig_z)
  } else {
    if (is.null(object$preprocessing$y_mean) || is.null(object$preprocessing$y_sd)) {
      warning("No target mean/sd stored; returning link-scale outputs.")
      out <- list(mu = mu_z, sigma = sig_z)
    } else {
      out <- list(
        mu    = object$preprocessing$y_mean + object$preprocessing$y_sd * mu_z,
        sigma = object$preprocessing$y_sd   * sig_z
      )
    }
  }

  # 4) Optional: Beiträge pro Feature (auf link-Skala, inkl. Baseline separat)
  if (isTRUE(return_contributions) && exists("forward_feature_net", mode = "function")) {
    p <- nrow(X_t)
    mu_contrib <- matrix(NA_real_, nrow = ncol(X_t), ncol = p)
    sigraw_contrib <- matrix(NA_real_, nrow = ncol(X_t), ncol = p)
    for (j in seq_len(p)) {
      resj <- forward_feature_net(matrix(X_t[j, ], nrow = 1), object$params, j,
                                  dropout_rate = 0, training = FALSE)
      mu_contrib[, j]     <- as.numeric(resj$output[1, ])
      sigraw_contrib[, j] <- as.numeric(resj$output[2, ])
    }
    out$contrib_mu_link <- mu_contrib
    out$contrib_sigraw_link <- sigraw_contrib
    out$beta_mu    <- object$params$beta_mu
    out$beta_sigma <- object$params$beta_sigma
  }

  out
}

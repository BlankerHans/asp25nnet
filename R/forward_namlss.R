forward_namlss <- function(X, params, dropout_rate = 0, training = FALSE) {

  arch <- attr(params, "architecture")
  n_features <- arch$n_features
  batch_size <- ncol(X)

  # Globaler Cache für alle Subnetze
  global_cache <- list()
  global_cache$n_features <- n_features
  global_cache$n_layers <- arch$n_layers

  mu_sum <- rep(0, batch_size)
  sigma_sum <- rep(0, batch_size)

  # Forward durch alle Feature-Netzwerke MIT CACHE
  for (j in 1:n_features) {
    x_j <- X[j, , drop = FALSE]

    # Forward mit Cache
    result <- forward_feature_net(x_j, params, j, dropout_rate, training)


    mu_sum <- mu_sum + result$output[1, ]
    sigma_sum <- sigma_sum + result$output[2, ]

    # Cache speichern
    global_cache[[paste0("subnet", j)]] <- result$cache
  }

  # globales beta addieren
  mu_raw <- mu_sum + params$beta_mu
  sigma_raw <- sigma_sum + params$beta_sigma

  # h^(k)
  mu <- mu_raw

  stability_const <- if (!is.null(params$sigma_floor)) params$sigma_floor else 0
  sigma <- Softplus(sigma_raw) + stability_const
  log_sigma <- log(sigma + 1e-8)

  global_cache$mu_raw <- mu_raw
  global_cache$sigma_raw <- sigma_raw
  global_cache$mu <- mu
  global_cache$sigma <- sigma
  global_cache$log_sigma <- log_sigma

  return(list(
    mu = mu,
    sigma = sigma,
    log_sigma = log_sigma,
    cache = global_cache
  ))
}

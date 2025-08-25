#' Forward Pass für ein einzelnes Feature-Netzwerk
#'
#' @param x_j Input für Feature j (1 x batch_size)
#' @param params Alle Modellparameter
#' @param j Feature-Index
#' @param dropout_rate Dropout-Rate (nur nach erster Schicht)
#' @param training Training-Modus
#' @return Matrix (2 x batch_size) mit Outputs für mu und sigma
forward_feature_net <- function(x_j, params, j, dropout_rate = 0, training = FALSE) {

  arch <- attr(params, "architecture")
  n_layers <- arch$n_layers

  # Cache für dieses Subnetz initialisieren
  cache_j <- list()
  cache_j$A0 <- x_j

  a <- x_j

  # Forward durch Hidden Layers
  for (l in 1:n_layers) {
    W_name <- paste0("W", j, "_", l)
    b_name <- paste0("b", j, "_", l)

    z <- params[[W_name]] %*% a + params[[b_name]]
    cache_j[[paste0("Z", l)]] <- z

    if (l < n_layers) {
      a <- ReLU(z)

      # Dropout nach erster Schicht (wie in Kneib Paper)
      if (training && dropout_rate > 0 && l == 1) {
        mask <- matrix(
          rbinom(length(a), 1, 1 - dropout_rate),
          nrow = nrow(a), ncol = ncol(a)
        )
        cache_j$dropout_mask <- mask
        a <- a * mask / (1 - dropout_rate)
      }
    } else {
      # Output Layer - keine Aktivierung hier
      a <- z
    }
    cache_j[[paste0("A", l)]] <- a
  }

  return(list(
    output = a,  # 2 x batch_size
    cache = cache_j  # Alle Zwischenergebnisse
  ))
}

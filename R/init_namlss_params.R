#' Initialize NAMLSS Parameters
#'
#' Initialisiert NAMLSS mit J Subnetzwerken (eins pro Feature),
#' jedes mit 2-dimensionalem Output (für mu und sigma)
#'
#' @param n_features Anzahl der Input-Features (J)
#' @param hidden_neurons Vector der Hidden Layer Größen, z.B. c(250, 50, 25)
#' @return Liste mit initialisierten Parametern für alle J Subnetzwerke
#' @export
init_namlss_params <- function(n_features, hidden_neurons = c(250, 50, 25)) {

  params <- list()

  for (j in 1:n_features) {
    # 1 Input -> hidden_neurons -> 2 Outputs (mu, sigma)
    layer_sizes <- c(1, hidden_neurons, 2)
    n_layers <- length(layer_sizes) - 1

    for (l in 1:n_layers) {
      W_name <- paste0("W", j, "_", l)
      b_name <- paste0("b", j, "_", l)

      # kaiming initialization für ReLU (aus original paper)
      fan_in <- layer_sizes[l]
      fan_out <- layer_sizes[l + 1]
      std_dev <- sqrt(2 / fan_in)

      params[[W_name]] <- matrix(
        rnorm(fan_out * fan_in, sd = std_dev),
        nrow = fan_out,
        ncol = fan_in
      )
      params[[b_name]] <- matrix(0, nrow = fan_out, ncol = 1)
    }
  }

  # Globale Bias-Terme (beta^(k))
  params$beta_mu <- 0
  params$beta_sigma <- 0


  attr(params, "architecture") <- list(
    n_features = n_features,
    n_h = hidden_neurons,
    n_layers = length(hidden_neurons)
  )

  return(params)
}

#' Initialize NAMLSS Parameters
#'
#' Initialisiert NAMLSS mit J Subnetzwerken (eins pro Feature),
#' jedes mit 2-dimensionalem Output (für mu und sigma)
#'
#' @param n_features Anzahl der Input-Features (J)
#' @param hidden_neurons Vector der Hidden Layer Größen, z.B. c(250, 50, 25)
#' @param y_mean mean of target (for numerical stability)
#' @param y_sd sd of target (for numerical stability)
#' @param sgima0 Initial guess for sigma (for numerical stability)
#' @return Liste mit initialisierten Parametern für alle J Subnetzwerke
#' @export
init_namlss_params <- function(n_features, hidden_neurons = c(250, 50, 25),
                               y_mean = 1, y_sd = 0, sigma0 = NULL, seed=42) {

  if (!is.null(seed)) set.seed(seed)

  params <- list()

  for (j in 1:n_features) {
    # 1 Input -> hidden_neurons -> 2 Outputs (mu, sigma)
    layer_sizes <- c(1, hidden_neurons, 2)
    n_layers <- length(layer_sizes) - 1

    for (l in 1:n_layers) {
      W_name <- paste0("W", j, "_", l)
      b_name <- paste0("b", j, "_", l)


      fan_in <- layer_sizes[l]
      fan_out <- layer_sizes[l + 1]

      if (l < n_layers) {
        # Hidden: Kaiming/He
        std_dev <- sqrt(2 / fan_in)
        params[[W_name]] <- matrix(rnorm(fan_out * fan_in, sd = std_dev),
                                   nrow = fan_out, ncol = fan_in)
        params[[b_name]] <- matrix(0, nrow = fan_out, ncol = 1)
      } else {
        # Output-Layer (2 x fan_in): Zeile1=μ-Head, Zeile2=σ-Head (vor Softplus)
        W <- matrix(0, nrow = 2, ncol = fan_in)
        b <- matrix(0, nrow = 2, ncol = 1)

        # μ-Gewichte sehr klein initialisieren (stabiler Start)
        W[1, ] <- rnorm(fan_in, sd = 1e-2)

        # σ-Gewichte bleiben 0 (neutral) – σ kommt anfangs nur über beta_sigma
        params[[W_name]] <- W
        params[[b_name]] <- b
      }
    }
  }

  # Globale Biases: μ startet am Mittel, σ startet bei σ0
  params$beta_mu <- y_mean
  s0 <- if (!is.null(sigma0)) sigma0 else max(y_sd, 1e-6)
  params$beta_sigma <- inv_softplus(s0)

  # Numerischer Schutz gegen σ≈0
  params$sigma_floor <- max(1e-6, 1e-3 * s0)


  attr(params, "architecture") <- list(
    n_features = n_features,
    n_h = hidden_neurons,
    n_layers = n_layers)

  return(params)
}

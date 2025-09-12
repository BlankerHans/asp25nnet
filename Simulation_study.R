install.packages("scoringRules")
library(scoringRules)
load_all()

# Szenario A: nichtlineares µ, konstantes σ
generate_data_A <- function(n) {
  x <- runif(n, -1, 1)                   # Feature
  mu <- sin(5 * x)                       # Erwartungswert
  sigma <- rep(1, n)                      # konstante Varianz
  eps <- rnorm(n)                         # Normalfehler
  y <- mu + sigma * eps                   # Zielvariable
  data.frame(x = x, y = y)
}


# Szenario B: nichtlineares µ, heteroskedastisches σ
generate_data_B <- function(n) {
  x <- runif(n, -1, 1)
  mu <- sin(5 * x)
  sigma <- 0.5 + 0.5 * plogis(x)          # σ hängt von x ab
  eps <- rnorm(n)
  y <- mu + sigma * eps
  data.frame(x = x, y = y)
}


# Szenario C: wie B, aber heavy tails (t-Verteilung)
generate_data_C <- function(n) {
  x <- runif(n, -1, 1)
  mu <- sin(5 * x)
  sigma <- 0.5 + 0.5 * plogis(x)
  eps <- rt(n, df = 3)                    # heavy tails
  y <- mu + sigma * eps
  data.frame(x = x, y = y)
}


# -------------------------------------
scenarios <- list(
  "A" = generate_data_A,
  "B" = generate_data_B,
  "C" = generate_data_C
)

# -------------------------------------
# Parameter
# -------------------------------------
n_reps <- 10   # Anzahl an Durchläufen pro Szenario
n <- 500      # Stichprobengröße

# -------------------------------------
# Hauptschleife
# -------------------------------------

results <- data.frame()

for (scn_name in names(scenarios)) {

  cat("Starte Szenario:", scn_name, "\n")
  gen_fun <- scenarios[[scn_name]]

  for (rep_id in 1:n_reps) {

    cat("  Replikation:", rep_id, "\n")
    set.seed(100 + rep_id)

    # Daten generieren
    data <- gen_fun(n)

    # Split
    data_split <- random_split(data["x"], normalization = FALSE)
    train_split <- data_split$train
    val_split <- data_split$validation
    test_split <- data_split$test
    targets <- data$y

    train_loader <- DataLoader(train_split)

    # Training
    model <- train(train_loader,targets , val_split, epochs = 10, optimizer = "adam")

    # forward pass auf Test set
    eval <- eval.NN(model, data_split, verbose = FALSE)
    loss <- eval$loss #reduction = mean
    rmse <- eval$rmse
    test_df_targets <- eval$test_df_targets

    #Berechne CRPS
    if (scn_name %in% c("A", "B")) {
      crps_values <- crps_norm(y = test_df_targets, location = eval$mu, scale = eval$sigma)
    } else if (scn_name == "C") {
      crps_values_lmls <- crps_t(y = test_df_targets, df = 3, location = eval$mu, scale = eval$sigma)
      cat("CRPS t dnn")
    }
      # mittlerer CRPS über Testset
    crps_mean <- mean(crps_values)

    # Ergebnisse für NN
    res <- data.frame(
      scenario = scn_name,
      rep      = rep_id,
      model    = "DNN",
      nll      = loss,
      rmse     = rmse,
      crps     = crps_mean
    )
    results <- dplyr::bind_rows(results, res) #füge neues res an bisherige results an

    # ----------- lmls ---------------------

    #Targets ziehen und an train_split dranbauen
    data_targets <- data$y
    train_targets <- data_targets[as.integer(rownames(train_split))]
    train_split <- cbind(train_split, train_targets)


    # Modell auf trainingsset fitten
    lmls <- lmls::lmls(train_targets ~ x, ~ x, data = train_split)

    # mu und sigma predicten
    mu_hat_lmls <- predict(lmls, type = "response", predictor = "location")
    sigma_hat_lmls <- predict(lmls, type = "response", predictor = "scale")


    # Kennzahlen auf Testset berechnen
    nll_lmls <- neg_log_lik(test_df_targets, mu_hat_lmls,
                             log(sigma_hat_lmls), reduction = "mean" )

    rmse_lmls <- sqrt(mean((mu_hat_lmls - test_df_targets)^2))


    if (scn_name %in% c("A", "B")) {
    crps_values_lmls <- crps_norm(y = test_df_targets, location = mu_hat_lmls, scale = sigma_hat_lmls)
    crps_mean_lmls <- mean(crps_values_lmls)
    }
    else if (scn_name == "C") {
      crps_values_lmls <- crps_t(y = test_df_targets, df = 3, location = mu_hat_lmls, scale = sigma_hat_lmls)
    cat("CRPS t lmls")
    }

    res_lmls <- data.frame(
      scenario = scn_name,
      rep = rep_id,
      model = "lmls",
      nll = nll_lmls,
      rmse = rmse_lmls,
      crps = crps_mean_lmls
    )
    results <- dplyr::bind_rows(results, res_lmls)

    }
}

# -------------------------------
# Ergebnisse
# -------------------------------

summary_results <- results |>
  dplyr::group_by(scenario, model) |>
  dplyr::summarise(
    mean_nll   = mean(nll),
    mean_rmse  = mean(rmse),
    mean_crps  = mean(crps),
    .groups = "drop"
) |>
  dplyr::rename(
  Scenario    = scenario,
  Model       = model,
  `Mean NLL`  = mean_nll,
  `Mean RMSE` = mean_rmse,
  `Mean CRPS` = mean_crps
)

# Export für LaTeX
knitr::kable(
  summary_results,
  format   = "latex",
  booktabs = TRUE,
  digits   = 3,
  caption  = "Simulation results across scenarios and models."
)



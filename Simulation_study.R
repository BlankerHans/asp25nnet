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
n_reps <- 3   # klein zum Testen
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
    data <- gen_fun(n)   # Dataframe mit x, y

    # Split
    data_split <- random_split(data["x"], normalization = FALSE)
    val_split <- data_split$validation
    test_split <- data_split$test
    targets <- data$y

    train_loader <- DataLoader(data_split$train)

    # Training
    model <- train(train_loader,targets , val_split, epochs = 10, optimizer = "adam")

    # forward pass auf Test set
    eval <- eval.NN(model, data_split, verbose = FALSE)
    loss <- eval$loss #reduction = mean
    rmse <- eval$rmse
    test_df_targets <- eval$test_df_targets

    #Berechne CRPS
    crps_values <- crps_norm(y = test_df_targets, mean = eval$mu, sd = eval$sigma)

    # mittlerer CRPS über Testset
    crps_mean <- mean(crps_values)

    # Ergebnisse sammeln
    res <- data.frame(
      scenario = scn_name,
      rep      = rep_id,
      nll      = loss,
      rmse     = rmse,
      crps     = crps_mean
    )
    results <- dplyr::bind_rows(results, res) #füge neues res an bisherige results an

    }
}

# -------------------------------
# Ergebnisse
# -------------------------------

summary_results <- results |>
  dplyr::group_by(scenario) |>
  dplyr::summarise(
    mean_nll   = mean(nll),
    mean_rmse  = mean(rmse),
    mean_crps  = mean(crps),
    .groups = "drop"
  )


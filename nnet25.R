library(lmls)
View(abdom)
library(numDeriv)


# Loss --------------------------------------------------------------------


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

x_i <- abdom$x
y_i <- abdom$y

model <- lm(y_i ~ x_i)

mu_i <- model$fitted.values

sigma_i <- summary(model)$sigma

sigma2_man <- t(model$residuals)%*%model$residuals / (length(model$residuals) - 1)

neg_log_lik(y_i, mu_i, sigma_i)


loss_vec <- neg_log_lik(y_i, mu_i, sigma_i, reduction = "raw")

library(ggplot2)
df <- data.frame(x = x_i, loss = loss_vec)
ggplot(df, aes(x = x, y = loss)) +
  geom_point(alpha = 0.6) +
  geom_smooth(se = FALSE) +
  labs(title = "Loss vs. x", x = "x_i", y = "Loss_i")



library(patchwork)

# Data‐Frame mit allen benötigten Spalten
df <- data.frame(
  x    = x_i,
  y    = y_i,
  loss = loss_vec
)

# 1) Plot: Loss vs. x
p1 <- ggplot(df, aes(x = x, y = loss)) +
  geom_point(alpha = 0.6) +
  geom_smooth(se = FALSE, color = "steelblue") +
  labs(title = "Loss with heteroskedasticity", x = "x_i", y = "Loss_i") +
  theme_minimal()

# 2) Plot: y vs. x
p2 <- ggplot(df, aes(x = x, y = y)) +
  geom_point(alpha = 0.6, color = "darkgray") +
  geom_smooth(method = "lm", se = FALSE, color = "tomato") +
  labs(title = "Heteroskedasticity within data", x = "x_i", y = "y_i") +
  theme_minimal()

# Plots nebeneinander anordnen
p1 + p2

x_hom <- rnorm(610)
y_hom <- 2*x_hom + rnorm(610, sd = 0.5)

model <- lm(y_hom ~ x_hom)

mu_i <- model$fitted.values

sigma_i <- summary(model)$sigma

loss_vec <- neg_log_lik(y_hom, mu_i, sigma_i, reduction = "raw")


df <- data.frame(
  x    = x_hom,
  y    = y_hom,
  loss = loss_vec
)

# 1) Plot: Loss vs. x
p1 <- ggplot(df, aes(x = x, y = loss)) +
  geom_point(alpha = 0.6) +
  geom_smooth(se = FALSE, color = "steelblue") +
  labs(title = "Loss with Homoskedasticity", x = "x_i", y = "Loss_i") +
  theme_minimal()

# 2) Plot: y vs. x
p2 <- ggplot(df, aes(x = x, y = y)) +
  geom_point(alpha = 0.6, color = "darkgray") +
  geom_smooth(method = "lm", se = FALSE, color = "tomato") +
  labs(title = "Homoskedasticity within data", x = "x_i", y = "y_i") +
  theme_minimal()

# Plots nebeneinander anordnen
p1 + p2
summary(model)


# Activation Functions ----------------------------------------------------

sigmoid <- function(x) {
  return(1 / (1 + exp(-x)))
}

ReLU <- function(x) {
  out <- pmax(0, x)
  cn <- colnames(x)
  dim(out) <- dim(x)  # Beibehaltung der Dimensionen
  colnames(out) <- cn
  return(out)
}

Softplus <- function(x) {
  return(log(1 + exp(x)))
}

ELU <- function(x, alpha = 1) {
  return(ifelse(x > 0, x, alpha * (exp(x) - 1)))
}

# LReLU
# ELU
# PReLU


# Data --------------------------------------------------------------------

random_split <- function(data, split=c(0.8, 0.2), normalization=TRUE) {

# prüft ob Split-Vektor zulässig ist und setzt 0.8,0.2 als default

  # 1) Typ / Länge prüfen
  if (!is.numeric(split) || length(split) != 2) {
    stop("`split` muss ein numerischer Vektor der Länge 2 sein, z.B. c(0.8, 0.2).")
  }
  # 2) Wertebereich prüfen
  if (any(split < 0) || any(split > 1)) {
    stop("Alle Einträge in `split` müssen zwischen 0 und 1 liegen.")
  }
  # 3) Summe prüfen
  if (sum(split) > 1) {
    stop("Die Summe von `split` darf maximal 1.0 sein (du hast ", sum(split), ").")
  }

  # Bestimmt Größe von Trainings- und Testdatensatz
  n      <- nrow(data)
  n_train <- floor(split[1] * n)
  n_test  <- floor(split[2] * n) # oder besser 1-n_train

  # normalization
  if (!is.logical(normalization) || length(normalization) != 1) {
    stop("`normalization` must be TRUE or FALSE")
  }
  if (normalization) {
    rn <- rownames(data)
    data <- scale(data)
    rownames(data) <- rn
  }

  # ohne shuffle
  train <- data[1:n_train, , drop = FALSE]
  test <- data[(n_train + 1):n, , drop = FALSE]

  # "shuffled" schon die daten
  # train <- sample(n, n_train)
  # test  <- sample(setdiff(seq_len(n), train), n_test)

  # return(list(
  #   train = data[train, , drop = FALSE],
  #   test  = data[test,  , drop = FALSE]
  # ))
  return(list(
    train = train,
    test  = test
  ))
}


datensplit <- random_split(abdom["x"])

targets <- abdom[["y"]]
train <- datensplit$train
test <- datensplit$test


train


DataLoader <- function(data, batch_size=32, shuffle=TRUE) {

#shuffelt Daten,erzeugt Start-Indizes für alle Batches (bei batch_size=32 : 1,33,65...)
#entnimmt für jeden Index i die Zeilen i bis i+batchsize-1 aus den Daten
#transponiert die Batches und returnt sie inkl Numerierung der enthaltenen data points

  data <- as.matrix(data)

  if (shuffle) {
    data <- data[sample(nrow(data)), , drop = FALSE]
  }

  n <- nrow(data)
  # Start-Indizes der Batches: 1, 1+batch_size, 1+2*batch_size, …
  starts <- seq(1, n, by = batch_size)

  # Für jeden Start einen kleinen Data Frame erzeugen
  batches <- lapply(starts, function(i) {
    mat <- data[i:min(i+batch_size-1, n), , drop = FALSE]
    mat_t <- t(mat)
    idx   <- as.integer(colnames(mat_t))
    list(
      batch   = mat_t,
      idx = idx
    )
  })

  return(batches)
}

batch_size <- 32

train_loader <- DataLoader(datensplit$train)

train_loader[[1]]$idx
train_loader[[1]]$batch

for (batch in train_loader) {
  print(batch$idx)
  print(batch$batch)
}

batch_counter <- 1

for (batch in train_loader){
  for (i in 1:length(batch$idx)) {
    print(paste("Batch:", batch_counter, "Index:", batch$idx[i], "x-Value", batch$batch[i]))
  }
  batch_counter <- batch_counter + 1
}



#  NNet -------------------------------------------------------------------

# shuffel data set (X and y)
# scale X
# create 80/20 train/test split of data set
# convert X and y to matrices and transpose

getLayerDimensions <- function(X, out_dim, hidden_neurons, train=TRUE) {
  n_x <- dim(X)[1]
  # X ist pxm mit P=feature anzahl und m=beobachtungen/batchsize (für abdom auch ein zeilenvektor)
  n_h <- hidden_neurons
  n_y <- out_dim

  dimensions_list <- list("n_x" = n_x,
               "n_h" = n_h,
               "n_y" = n_y)

  return(dimensions_list)
}

# get targets with batch$idx
# targets[train_loader[[1]]$idx]

dimensions <- getLayerDimensions(train_loader[[1]]$batch, 2, hidden_neurons = 3)
dimensions$n_h




init_params <- function(dimensions_list, seed=42) {
  set.seed(seed)
  list(
    W1 = matrix(rnorm(dimensions_list$n_h * dimensions_list$n_x, sd = 0.1), nrow = dimensions_list$n_h, ncol = dimensions_list$n_x),
    b1 = matrix(0, nrow = dimensions_list$n_h, ncol = 1),
    W2 = matrix(rnorm(dimensions_list$n_y * dimensions_list$n_h, sd = 0.1), nrow = dimensions_list$n_y, ncol = dimensions_list$n_h),
    b2 = matrix(0, nrow = dimensions_list$n_y, ncol = 1)

    #Dimensionen:
    # W1: n_h x n_x
    # b1: n_h x 1
    # W2: n_y x n_h
    # b2: n_y x 1

  )
}

test_params <- init_params(dimensions)




forward_onehidden <- function(X, params) {
  ones <- matrix(1, nrow = 1, ncol = dim(X)[2]) # oder in init params direkt für b1&b2 eine kxb matrix generieren mit selben biasen?
  Z1 <- params$W1 %*% X + params$b1 %*% ones
  A1 <- ReLU(Z1)
  Z2 <- params$W2 %*% A1 + params$b2 %*% ones #erste Zeile: my, zweite Zeile: log_sigma
  mu_hat  <- Z2[1, , drop = FALSE]
  log_sigma_hat <- Z2[2, , drop = FALSE]

  #Dimensionen:
  #Z1: n_h x m
  #Z2: n_y x m

  cache <- list("Z1" = Z1,
                "A1" = A1,
                "Z2" = Z2,
                "mu" = mu_hat,
                "log_sigma" = log_sigma_hat)

  return(cache)
}

forward_test <- forward_onehidden(train_loader[[1]]$batch, test_params)





# Backpropagation für ein 1-Hidden-Layer-Netz
backprop_onehidden <- function(X, y, cache, params) {
  #
  # X       : p × m Input-Matrix (Features × Batchsize)
  # y       : Vektor der Länge m mit Zielwerten
  # cache   : Liste mit Forward-Ergebnissen Z1, A1, Z2, mu, log_sigma
  # params  : Liste mit Parametern W1, b1, W2, b2
  #
  m <- ncol(X)

  # 1) Berechne die Ableitungen der Loss nach den Output-Preactivations Z2
  mu        <- as.numeric(cache$mu)         # Länge m
  log_sigma <- as.numeric(cache$log_sigma)  # Länge m
  sigma2    <- exp(2 * log_sigma)           # σ^2

  # δ_mu = dL/dz_mu = (mu - y) / σ^2
  delta_mu  <- (mu - y) / sigma2            # Länge m

  # δ_eta = dL/dz_eta = 1 - (y - mu)^2 / σ^2
  delta_eta <- 1 - (y - mu)^2 / sigma2      # Länge m

  # δ2 als 2×m Matrix
  delta2 <- rbind(delta_mu,
                  delta_eta)

  # 2) Gradienten für W2 und b2
  dW2 <- (delta2 %*% t(cache$A1)) / m       # 2×k
  db2 <- rowSums(delta2) / m                # Länge 2

  # 3) Rückprop in Hidden Layer
  #    dA1 = W2^T %*% δ2
  dA1    <- t(params$W2) %*% delta2         # k×m

  #    dZ1 = dA1 * ReLU'(Z1), ReLU'(z)=1[z>0]
  dZ1    <- dA1 * (cache$Z1 > 0)            # k×m

  # 4) Gradienten für W1 und b1
  dW1 <- (dZ1 %*% t(X)) / m                 # k×p
  db1 <- rowSums(dZ1) / m                   # Länge k

  # 5) Gib alle Gradienten zurück
  list(
    dW1 = dW1,
    db1 = matrix(db1, ncol = 1),
    dW2 = dW2,
    db2 = matrix(db2, ncol = 1)
  )
}




train_network <- function(train_loader, targets, dimensions,
                          epochs = 100, lr = 0.01) {
  params  <- init_params(dimensions)
  history <- numeric(epochs)

  for (e in seq_len(epochs)) {
    batch_losses <- numeric(length(train_loader))

    for (i in seq_along(train_loader)) {
      Xb <- train_loader[[i]]$batch
      yb <- targets[ train_loader[[i]]$idx ]

      fwd <- forward_onehidden(Xb, params)
      # Mean‐Loss pro Batch
      batch_losses[i] <- neg_log_lik(yb,
                                     as.numeric(fwd$mu),
                                     as.numeric(fwd$log_sigma),
                                     reduction = "mean")

      grads <- backprop_onehidden(Xb, yb, fwd, params)
      params$W1 <- params$W1 - lr * grads$dW1
      params$b1 <- params$b1 - lr * grads$db1
      params$W2 <- params$W2 - lr * grads$dW2
      params$b2 <- params$b2 - lr * grads$db2
    }

    # Durchschnitt der Batch‐Means = Epoch‐Mean
    history[e] <- mean(batch_losses)
    message(sprintf("Epoch %3d/%d – Loss: %.6f",
                    e, epochs, history[e]))
  }

  list(params = params, history = history)
}


res <- train_network(train_loader,
                     targets,
                     dimensions,
                     epochs = 10000,
                     lr = 0.001)

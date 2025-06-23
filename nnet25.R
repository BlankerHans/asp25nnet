library(lmls)
View(abdom)
library(numDeriv)


# Loss --------------------------------------------------------------------


neg_log_lik <- function(y, mu, sigma, reduction = c("sum","mean","raw")) {
  
  reduction <- match.arg(reduction)
  
  loss_i <- log(sigma) + (y - mu)^2 / (2 * sigma^2)
  
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
  return(pmax(0, x))
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

random_split <- function(data, split=c(0.8, 0.2)) {
  
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
  
  n      <- nrow(data)
  n_train <- floor(split[1] * n)
  n_test  <- floor(split[2] * n) # oder besser 1-n_train
  
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


datensplit <- random_split(abdom)

train <- datensplit$train
test <- datensplit$test


DataLoader <- function(data, batch_size=32, shuffle=TRUE) {
  
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



#  NNet -------------------------------------------------------------------

# shuffel data set (X and y)
# scale X
# create 80/20 train/test split of data set
# convert X and y to matrices and transpose

getLayerSize <- function(X, y, hidden_neurons, train=TRUE) {
  n_x <- dim(X)[1]
  n_h <- hidden_neurons
  n_y <- dim(y)[1]   
  
  size <- list("n_x" = n_x,
               "n_h" = n_h,
               "n_y" = n_y)
  
  return(size)
}





# NN functions -----------------------------------------------------------

init_params <- function(size) {
  set.seed(42)
  list(
    W1 = matrix(rnorm(size$n_h * size$n_x, sd = 0.1), nrow = size$n_h, ncol = size$n_x),
    b1 = matrix(0, nrow = size$n_h, ncol = 1),
    W2 = matrix(rnorm(size$n_y * size$n_h, sd = 0.1), nrow = size$n_y, ncol = size$n_h),
    b2 = matrix(0, nrow = size$n_y, ncol = 1)
  )
}

forward_onehidden <- function(X, params) {
  ones <- matrix(1, nrow = 1, ncol = dim(X)[2])
  Z1 <- params$W1 %*% X + params$b1 %*% ones
  A1 <- ReLU(Z1)
  Z2 <- params$W2 %*% A1 + params$b2 %*% ones
  z_mu  <- Z2[1, , drop = FALSE]
  z_eta <- Z2[2, , drop = FALSE]
  mu_hat <- z_mu
  eta_hat <- Softplus(z_eta)
  sigma_hat <- exp(eta_hat)
  cache <- list(X = X, Z1 = Z1, A1 = A1, Z2 = Z2,
                mu = mu_hat, eta = eta_hat, sigma = sigma_hat)
  list(mu_hat = mu_hat, sigma_hat = sigma_hat, cache = cache)
}

loss_nll <- function(y, mu, sigma) {
  sum(log(sigma) + (y - mu)^2 / (2 * sigma^2))
}

backward_onehidden <- function(y, params, cache) {
  m <- ncol(y)
  mu_hat <- cache$mu
  sigma_hat <- cache$sigma
  z_eta <- cache$Z2[2, , drop = FALSE]

  delta_mu  <- -(y - mu_hat) / sigma_hat^2
  delta_eta <- (1 - (y - mu_hat)^2 / sigma_hat^2) * sigmoid(z_eta)

  delta2 <- rbind(delta_mu, delta_eta)
  dW2 <- delta2 %*% t(cache$A1)
  db2 <- delta2 %*% matrix(1, nrow = m, ncol = 1)

  delta1_raw <- t(params$W2) %*% delta2
  delta1 <- delta1_raw * (cache$Z1 > 0)

  dW1 <- delta1 %*% t(cache$X)
  db1 <- delta1 %*% matrix(1, nrow = m, ncol = 1)

  list(dW1 = dW1, db1 = db1, dW2 = dW2, db2 = db2)
}

update_params <- function(params, grads, lr = 0.01) {
  params$W1 <- params$W1 - lr * grads$dW1
  params$b1 <- params$b1 - lr * grads$db1
  params$W2 <- params$W2 - lr * grads$dW2
  params$b2 <- params$b2 - lr * grads$db2
  params
}

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
  
  #Bestimmt Größe von Trainings- und Testdatensatz 
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






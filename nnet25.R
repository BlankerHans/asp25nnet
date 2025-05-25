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

# LReLU
# ELU
# PReLU


# Data --------------------------------------------------------------------

random_split <- function(data, split=c(0.8, 0.2)) {
  split <- match.arg(split)
  
  n <- nrow(data)
  train_indices <- sample(1:n, size = floor(n * split[1]))
  
  train_data <- data[train_indices, ]
  test_data <- data[-train_indices, ]
  
  return(list(train = train_data, test = test_data))
}

DataLoader <- function(data, batch_size=32, shuffle=TRUE) {
  if (shuffle) {
    data <- data[sample(nrow(data)), ]
  }
  
  n <- nrow(data)
  batches <- split(data, rep(1:ceiling(n / batch_size), each=batch_size, length.out=n))
  
  return(batches)
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




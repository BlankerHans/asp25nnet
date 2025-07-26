load_all()

set.seed(42)
n <- 500
beta <- 2
sigma0 <- 0.5

# 1) Simuliere x
x <- abs(rnorm(n, mean = 0, sd = 1))

# 2) Berechne für jede x_i die Fehler‐Std-Dev
sigma_x <- sigma0 * x

# 3) Ziehe heteroskedastische Fehler
eps <- rnorm(n, mean = 0, sd = sigma_x)

# 4) Erzeuge y
y <- beta * x + eps

# Kurzer Blick auf Varianz in Abhängigkeit von x
plot(x, y,
     xlab = "x",
     ylab = "y",
     main = "Heteroskedastie")

dataframe <- as.data.frame(cbind(x, y))
View(dataframe)
colnames(dataframe) <- c("unab", "abh")

split <- train_val_test(dataframe["unab"])
train <- split$train
val <- split$val
test <- split$test


nrow(dataframe) == nrow(test)+nrow(train)+nrow(val)

train_loader <- DataLoader(split$train)

# val_loader <- DataLoader(split$val, shuffle=FALSE) bräuchte man nur wenn wir das
# training umschreiben so dass wir auch pro batch einen validation forward machen




targets <- dataframe$abh
val_targets <- targets[as.integer(rownames(val))]

dimensions <- getLayerDimensions(train_loader[[1]]$batch, 2, hidden_neurons = 3)




train_network_val_adam(train_loader, targets, dimensions, t(val), val_targets)


train_network(train_loader,
              targets,
              dimensions)



params <- init_params(dimensions)
W1 <- params$W1
params$W2

dim(W1)
dim(train_loader[[1]]$batch)
class(W1)
class(train_loader[[1]]$batch)
dim(W1%*% train_loader[[1]]$batch)


model <- train(train_loader, targets, dimensions, t(val), val_targets, optimizer = "adam")
model

# oder ggplot nutzen?

plot(1:length(model$train_loss), model$train_loss, type = "l", col = "blue")
plot(1:length(model$val_loss), model$val_loss, type = "l", col = "red")

summary(model)


View(abdom)
abdom_split <- train_val_test(abdom['x'], normalization=FALSE)
abdom_targets <- abdom$y

train_abdom <- abdom_split$train
val_abdom <- abdom_split$validation
val_abdom_targets <- abdom_targets[as.integer(rownames(val_abdom))]


abdom_loader <- DataLoader(train_abdom)
dimensions <- getLayerDimensions(abdom_loader[[1]]$batch, 2, hidden_neurons = 50)

model2 <- train(abdom_loader, abdom_targets, dimensions, t(val_abdom), val_abdom_targets, optimizer="adam", epochs=1000)
model2
summary(model2)


fwd_abdom <- forward_onehidden(t(abdom['x']), model2$params)
mu <- fwd_abdom$mu
plot(abdom$x, abdom$y, xlab = "x", ylab = "y", main = "Abdomen Data")
lines(abdom$x, mu, col = "red", lwd = 2)
sigma <- exp(fwd_abdom$log_sigma)
upper <- mu + 1.96 * sigma
lower <- mu - 1.96 * sigma

polygon(
  c(abdom$x, rev(abdom$x)),
  c(upper, rev(lower)),
  col = rgb(0.2, 0.2, 1, alpha = 0.2),
  border = NA
)



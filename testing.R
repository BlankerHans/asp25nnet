data <- abdom

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
val_loader <- DataLoader(split$val, shuffle=FALSE)

targets <- dataframe$abh
val_targets <- targets[val_loader$idx]

dimensions <- getLayerDimensions(train_loader[[1]]$batch, 2, hidden_neurons = 3)

train_network_val_adam()

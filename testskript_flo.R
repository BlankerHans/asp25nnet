# Summary function test without normalization
load_all()
data <- abdom
abdom_split <- random_split(abdom['x'], normalization=FALSE)
abdom_targets <- abdom$y

train_abdom <- abdom_split$train
val_abdom <- abdom_split$validation


abdom_loader <- DataLoader(train_abdom, batch_size = 256)

model <- train(abdom_loader, abdom_targets,
               #val_abdom,
               hidden_neurons = c(50), optimizer="adam", epochs=500, lr=0.01)
model <- train(abdom_loader, abdom_targets,hidden_neurons = c(50), optimizer="adam", epochs=500, lr=0.01)

#summary.NN(model, data, "y", yscale="robust", drop_first=10)
eval.NN(model, abdom_split)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Summary function test with normalization
load_all()
data <- abdom
abdom_split <- random_split(abdom['x'], normalization=TRUE)
abdom_targets <- abdom$y
norm <- abdom_split$normalization_params

train_abdom <- abdom_split$train
val_abdom <- abdom_split$validation
val_abdom_targets <- abdom_targets[as.integer(rownames(val_abdom))]


abdom_loader <- DataLoader(train_abdom, batch_size = 256)

model <- train(abdom_loader, abdom_targets, t(val_abdom), val_abdom_targets, c(50), optimizer="adam", epochs=500, lr=0.01, normalization_params = norm)
summary.NN(model, data, "y", yscale="robust", drop_first=10)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Summary function test with random generated dataset

set.seed(123)  # für Reproduzierbarkeit
n <- 200
x <- seq(3, 50, length.out = n)
mu_gen <- sin(x) + 0.3 * x^2
sigma_gen <- 0.2 + 0.3 * abs(x)
y <- mu_gen + rnorm(n, mean = 0, sd = sigma_gen)
df <- data.frame(x = x, y = y)
plot(df$x, df$y, main = "Heteroskedastic Nonlinear Data",
     xlab = "x", ylab = "y", pch = 19, col = rgb(0,0,1,0.5))


df_split <- random_split(df['x'], normalization=FALSE)
df_targets <- df$y
norm <- df_split$normalization_params

train_df <- df_split$train
val_df <- df_split$validation
val_df_targets <- df_targets[as.integer(rownames(val_df))]


df_loader <- DataLoader(train_df, batch_size = 256)

model <- train(df_loader, df_targets, t(val_df), val_df_targets, c(50), optimizer="adam", epochs=500, lr=0.01)
summary.NN(model, df, "y", yscale="robust", drop_first=10)

#~~~~~~~~~~~~~~Summary test with 2 Input dataset~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
load_all()
set.seed(42)

n <- 500  # Anzahl der Datenpunkte

# Zwei Input-Variablen
x1 <- runif(n, -5, 5)
x2 <- rnorm(n, mean = 0, sd = 2)

# Heteroskedastisches Rauschen (variiert mit x1 und x2)
sigma <- 0.5 + 0.3 * abs(x1) + 0.2 * (x2^2)

# Nichtlineare Abhängigkeit für y
y <- sin(x1) + 0.5 * x2^2 + rnorm(n, mean = 0, sd = sigma)

# Datensatz zusammenfassen
df_whole <- data.frame(x1 = x1, x2 = x2, y = y)
df_x <- data.frame(x1 = x1, x2 = x2)
df_split <- random_split(df_x, normalization=TRUE)
df_targets <- df_whole$y


train_df <- df_split$train
val_df <- df_split$validation
val_df_targets <- df_targets[as.integer(rownames(val_df))]


df_loader <- DataLoader(train_df, batch_size = 256)

model <- train(df_loader, df_targets, t(val_df), val_df_targets, c(50), optimizer="adam", epochs=500, lr=0.01)
load_all()
summary.NN(model, df_whole, "y", yscale="robust", drop_first=10)
rglwidget()
#-------------------------------------------------------------

set.seed(42)
n     <- 1000
x     <- runif(n, 0, 10)
mu    <- 5 * sin(x)
sigma <- 0.5 + 0.3 * x
eps   <- rnorm(n, 0, sigma)
y     <- mu + eps
df    <- data.frame(x = x, y = y, mu = mu, sigma = sigma)

ord   <- order(df$x)
plot(df$x, df$y, pch = 16, cex = 0.6, xlab = "x", ylab = "y", main = "Nicht‐linear + Heteroskedastisch")


View(df)
sim_split <- random_split(df['x'], normalization=TRUE)
sim_targets <- df$y

train_sim <- sim_split$train
val_sim <- sim_split$validation
val_sim_targets <- sim_targets[as.integer(rownames(val_sim))]


sim_loader <- DataLoader(train_sim, batch_size = 32)


model3 <- train(sim_loader, sim_targets, t(val_sim), val_sim_targets, c(50),optimizer="adam", epochs=1000, lr=0.01)
class(model3)
summary.NN(model3, show_plot=TRUE, yscale="robust", drop_first=10)

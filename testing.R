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

split <- random_split(dataframe["unab"])
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


model <- train_variable(train_loader, targets, dimensions, t(val), val_targets, c(50), optimizer = "adam")
model

# oder ggplot nutzen?

#plot(1:length(model$train_loss), model$train_loss, type = "l", col = "blue")
#plot(1:length(model$val_loss), model$val_loss, type = "l", col = "red")

summary(model)


View(abdom)
abdom_split <- random_split(abdom['x'], normalization=FALSE)
abdom_targets <- abdom$y

train_abdom <- abdom_split$train
val_abdom <- abdom_split$validation
val_abdom_targets <- abdom_targets[as.integer(rownames(val_abdom))]


abdom_loader <- DataLoader(train_abdom, batch_size = 256)

model2 <- train_variable(abdom_loader, abdom_targets, t(val_abdom), val_abdom_targets, c(50), optimizer="adam", epochs=2000, lr=0.01)
model2
summary.NN(model2, yscale="robust", drop_first=10)


# folgendes in summary übertragen
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



# Non-linear data & heteroskedasticity ------------------------------------


set.seed(42)
n     <- 1000
x     <- runif(n, 0, 10)
mu    <- 5 * sin(x)
sigma <- 0.5 + 0.3 * x
eps   <- rnorm(n, 0, sigma)
y     <- mu + eps
df    <- data.frame(x = x, y = y)

ord   <- order(df$x)
plot(df$x, df$y, pch = 16, cex = 0.6, xlab = "x", ylab = "y", main = "Nicht‐linear + Heteroskedastisch")


#View(df)
sim_split <- random_split(df['x'], normalization=FALSE)
sim_targets <- df$y

train_sim <- sim_split$train
val_sim <- sim_split$validation
val_sim_targets <- sim_targets[as.integer(rownames(val_sim))]


sim_loader <- DataLoader(train_sim, batch_size = 32)


model3 <- train(sim_loader, sim_targets, t(val_sim), val_sim_targets, c(50),optimizer="adam", epochs=1000, lr=0.01)
class(model3)
summary.NN(model3, df, "y", show_plot=TRUE, yscale="auto", drop_first=0)



# fwd_sim <- forward(t(df['x']), model3$params)
mu_sim <- fwd_sim$mu
sigma_sim <- exp(fwd_sim$log_sigma)

# Sortierindex berechnen
ord <- order(df$x)

plot(df$x, df$y,
     pch   = 16,
     cex   = 0.6,
     xlab  = "x",
     ylab  = "y",
     main  = "Nicht‐linear + Heteroskedastisch")


lines(df$x[ord], mu_sim[ord],
      col = "red",
      lwd = 2)


upper <- mu_sim + 1.96 * sigma_sim
lower <- mu_sim - 1.96 * sigma_sim
polygon(
  x    = c(df$x[ord], rev(df$x[ord])),
  y    = c(upper[ord], rev(lower[ord])),
  col   = rgb(1, 0, 0, alpha = 0.2),
  border = NA
)



# Test for variable layer sizes and multiple inputs
multi_layer_dims <- getLayerDimensions_variable(sim_loader[[1]]$batch, out_dim = 2, hidden_neurons = c(10, 5, 10))
dim(sim_loader[[1]]$batch)[1]

params <- init_params_variable(multi_layer_dims)
lapply(params, dim)


forward_variable(sim_loader[[1]]$batch, params)






# Testing NAMLSS ----------------------------------------------------------

# NAM vs NN on syntethic heteros. & non-linearity data
nam <- train_namls(sim_loader, sim_targets, 1,  c(50), t(val_sim), val_sim_targets,
                    optimizer="adam", epochs=3000, lr=0.001,
                    dropout_rate=0, lr_decay=0.95, lr_patience=100)

#View(df)
df_var <- df[, c("x", "y")]

test <- summary.NAMLSS(nam,
        data = df_var,             # DataFrame mit x-Spalten + Zielspalte
        target_col = "y",      # Name der Zielspalte
        show_plot = TRUE,
        yscale = "robust",       # "auto" | "log" | "robust"
        cap_quantile = 0.99,
        drop_first = 10,
        feature_plots = TRUE,  # partielle Effektplots bei >1 Features
        max_features = 6,
        ci_z = 1.96)

# California Housing Data
# ca_housing[ , setdiff(names(ca_housing), "target")] alles außer target

# reduced_df <- ca_housing[, c("MedInc", "HouseAge", "AveRooms", "Population", "target"), drop = FALSE]
reduced_df <- ca_housing[, c("MedInc", "target"), drop = FALSE]
input_vars <- reduced_df[, setdiff(names(reduced_df), "target"), drop = FALSE]

ca_housing_split <- random_split(input_vars, normalization=TRUE)

# median house value for California districts, in $100,000
targets_ca_housing <- ca_housing$target

train_ca_housing <- ca_housing_split$train

val_ca_housing <- ca_housing_split$validation
# val_targets vielleicht noch automatisch erkennen mit in train aufnehmen!?
val_targets_ca_housing <- targets_ca_housing[as.integer(rownames(val_ca_housing))]

ca_housing_loader <- DataLoader(train_ca_housing, batch_size = 1024)

nam_housing <- train_namls(ca_housing_loader, targets_ca_housing, 1,  c(50), t(val_ca_housing), val_targets_ca_housing,
                    optimizer="adam", epochs=2000, lr=0.001,
                    dropout_rate=0.1, lr_decay=0.95, lr_patience=10)
summary.NAMLS(nam_housing,
               data = reduced_df,             # DataFrame mit x-Spalten + Zielspalte
               target_col = "c",      # Name der Zielspalte
               show_plot = TRUE,
               yscale = "robust",       # "auto" | "log" | "robust"
               cap_quantile = 0.99,
               drop_first = 1,
               feature_plots = TRUE,  # partielle Effektplots bei >1 Features
               max_features = 2,
               ci_z = 1.96)



# Insurance Data ----------------------------------------------------------


# insurance[ , setdiff(names(insurance), "charges")] alles außer target

# reduced_df <- ca_housing[, c("MedInc", "HouseAge", "AveRooms", "Population", "target"), drop = FALSE]
reduced_df <- insurance[, c("age", "charges"), drop = FALSE]
input_vars <- reduced_df[, setdiff(names(reduced_df), "charges"), drop = FALSE]

insurance_split <- random_split(input_vars, normalization=TRUE)

# charges => target
targets_insurance <- insurance$charges

train_insurance <- insurance_split$train

val_insurance <- insurance_split$validation
# val_targets vielleicht noch automatisch erkennen mit in train aufnehmen!?
val_targets_insurance <- targets_insurance[as.integer(rownames(val_insurance))]

insurance_loader <- DataLoader(train_insurance, batch_size = 128)

dnn_insurance <- train(insurance_loader, targets_insurance,
                       t(val_insurance), val_targets_insurance,
                       c(32),optimizer="adam", epochs=10000, lr=0.001)
summary.NN(dnn_insurance,
           data = reduced_df,             # DataFrame mit x-Spalten + Zielspalte
           target_col = "charges",      # Name der Zielspalte
           show_plot = TRUE,
           yscale = "robust",       # "auto" | "log" | "robust"
           cap_quantile = 0.99,
           drop_first = 1)





nam_insurance <- train_namls(insurance_loader, targets_insurance, 1,
                              c(50), t(val_insurance), val_targets_insurance,
                            optimizer="adam", epochs=5000, lr=0.001,
                            dropout_rate=0, lr_decay=0.99, lr_patience=10)
summary.NAMLSS(nam_insurance,
               data = reduced_df,             # DataFrame mit x-Spalten + Zielspalte
               target_col = "target",      # Name der Zielspalte
               show_plot = TRUE,
               yscale = "robust",       # "auto" | "log" | "robust"
               cap_quantile = 0.99,
               drop_first = 1,
               feature_plots = FALSE,  # partielle Effektplots bei >1 Features
               max_features = 4,
               ci_z = 1.96)


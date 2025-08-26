# Modellgüte:

# Coverage der 95%-CI
# X_val: (features x N_val)
fwd <- forward_namlss(t(as.matrix(val_ca_housing[, c("HouseAge","AveRooms")])),
                      nam_housing$params, dropout_rate = 0, training = FALSE)
mu    <- as.numeric(fwd$mu)
sigma <- as.numeric(fwd$sigma)
y     <- val_targets_ca_housing
cover <- mean( (y >= mu - 1.96*sigma) & (y <= mu + 1.96*sigma) )
print(cover)  # sollte nahe 0.95 liegen (bei Normalannahme)


# Standardisierte Residuen sollten ~N(0,1) sein

z <- (y - mu) / sigma
# Grober Test:
mean(z); sd(z)   # ~0 und ~1
# QQ-Plot:
qqnorm(z); qqline(z)


#RMSE (nur um Gefühl für den Punkt-Fit zu kriegen)

rmse <- sqrt(mean((y - mu)^2))
print(rmse)



# nam indize check --------------------------------------------------------



# 1) Hat jeder Batch gültige Indizes?
all(sapply(ca_housing_loader, function(b) all(!is.na(b$idx))))

# 2) Stimmen Targets zu den Indizes? (sollte TRUE sein)
i <- sample(length(ca_housing_loader), 1)
idx <- ca_housing_loader[[i]]$idx
identical(targets_ca_housing[idx], ca_housing$target[idx])

# 3) Ist μ wirklich (nahe) konstant?
Xplot <- t(as.matrix(reduced_df[, c("MedInc","HouseAge","AveRooms","Population")]))
fwd   <- forward_namlss(Xplot, nam_housing$params, dropout_rate=0, training=FALSE)
sd_mu <- sd(as.numeric(fwd$mu));  cor_mu_y <- cor(as.numeric(fwd$mu), reduced_df$target)
c(sd_mu = sd_mu, cor_mu_y = cor_mu_y)


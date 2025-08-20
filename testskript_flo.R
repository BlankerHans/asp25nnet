load_all()
data <- abdom

colnames(data) <- c("unab", "abh")

split <- random_split(data["unab"])
train <- split$train
val <- split$val
test <- split$test

train_loader <- DataLoader(split$train)

targets <- data$abh
val_targets <- targets[as.integer(rownames(val))]

dimensions <- getLayerDimensions(train_loader[[1]]$batch, 2, hidden_neurons = 3)
model <- train(train_loader, targets, dimensions, t(val), val_targets,epochs = 50,lr = 0.001 ,optimizer = "sgd")
summary.NN(model)



load_all()
abdom_split <- random_split(abdom['x'], normalization=FALSE)
abdom_targets <- abdom$y

train_abdom <- abdom_split$train
val_abdom <- abdom_split$validation
val_abdom_targets <- abdom_targets[as.integer(rownames(val_abdom))]


abdom_loader <- DataLoader(train_abdom, batch_size = 256)

model <- train(abdom_loader, abdom_targets, t(val_abdom), val_abdom_targets, c(50), optimizer="adam", epochs=500, lr=0.01)
summary.NN(model, yscale="robust", drop_first=10)



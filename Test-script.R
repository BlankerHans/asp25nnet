#~~~~~~~~~~~~~~~~~TEST script~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
library(lmls)

set.seed(42)

datensplit <- random_split(abdom["x"])

targets <- abdom[["y"]]
train <- datensplit$train
test <- datensplit$test


train_loader <- DataLoader(datensplit$train)
#  NNet -------------------------------------------------------------------



dimensions <- getLayerDimensions(train_loader[[1]]$batch, 2, hidden_neurons = 3)

res <- train_network(train_loader,
                     targets,
                     dimensions,
                     epochs = 100,
                     lr = 0.001)

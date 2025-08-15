# asp25nnet

`asp25nnet` bietet grundlegende Werkzeuge zum Aufbau und Training eines einfachen neuronalen Netzes mit einer versteckten Schicht in R.

## Installation

```r
# install.packages("devtools")
devtools::install_github("owner/asp25nnet")
```

## Beispiel

```r
library(asp25nnet)

# Beispiel-Daten
set.seed(1)
df <- data.frame(x = rnorm(100), y = rnorm(100))

# Aufteilen und Batches erzeugen
splits <- random_split(df)
train_loader <- DataLoader(splits$train)

# Netzwerkdimensionen festlegen und trainieren
dims <- getLayerDimensions(train_loader[[1]]$batch, out_dim = 2, hidden_neurons = 10)
model <- train(train_loader, df$y, dims, optimizer = "adam", epochs = 5)
```

## Wichtige Funktionen

- `random_split()` – teilt Daten in Trainings-, Validierungs- und Testmengen
- `DataLoader()` – erstellt Batches aus den Trainingsdaten
- `getLayerDimensions()` – berechnet Eingabe-, Hidden- und Ausgabedimensionen
- `train()` – trainiert das Netz (SGD oder Adam)
- `forward_onehidden()` – führt einen Forward-Pass zur Vorhersage aus

---

Weitere Notizen: https://www.notion.so/nnet25-1fc5a72b4e5b80d6a1f7db0320fb5149?pvs=4

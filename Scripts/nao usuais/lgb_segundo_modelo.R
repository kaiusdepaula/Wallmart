library(data.table)
library(lightgbm)
library(ggplot2)

#Leitura dos dados
dataset <- as.data.table(arrow::read_parquet("~/Wallmart/dados_de_treinamento/data_processada_CA_2.parquet"))

#Retirando o periodo usado para prever
dataset <- dataset[date <= as.Date("2016-06-19") - 28L]

#Declarando tipo
col_categoricas <- c(
  "dept_id", "cat_id",
  "item_id", "event_name_1", "event_name_2", "event_type_2",
  "event_type_1", "snap_CA"
)
dataset[, (col_categoricas) := lapply(.SD, as.factor), .SDcols = col_categoricas]

#Colunas que vão sair do escopo de treinamento 
xFEAT <- setdiff(names(dataset), c(
  "date", "item_id", "snap_TX", "snap_WI", "sales", "store_id"
))

#Lista vazia para captar as métricas de validação
plot_data <- data.table()

#Lista de parâmetros de treinamento do modelo
param <- list(boosting_type = "gbdt",
              objective = "tweedie",
              tweedie_variance_power = 1.1,
              metric = "rmse",
              subsample = 0.5,
              subsample_freq = 1,
              learning_rate = 0.015,
              num_leaves = 2**8-1,
              min_data_in_leaf = 2**8-1,
              feature_fraction = 0.5,
              max_bin = 100,
              n_estimators = 3500,
              boost_from_average = FALSE,
              verbose = -1,
              seed = 1995)


#Treinamento
dt_train <- dataset[date < as.Date("2016-05-22") - 56L]

dt_trainMT = as.matrix(dt_train[,lapply(.SD, as.numeric),.SDcols = xFEAT])
dt_trainDM = lgb.Dataset(dt_trainMT, label = log1p(dt_train[["sales"]]))

#Validação
dt_valid <- dataset[date >= as.Date("2016-05-22") - 56L & date < as.Date("2016-05-22") - 28L]

dt_validMT = as.matrix(dt_valid[,lapply(.SD, as.numeric),.SDcols = xFEAT])
dt_validDM = list(test = lgb.Dataset.create.valid(dt_trainDM, dt_validMT, label = log1p(dt_valid[["sales"]])))


#Conjunto de teste
dt_test <- dataset[date > as.Date("2016-05-22") - 28L]
rm(dataset)
invisible(gc())

dt_testMT = as.matrix(dt_test[,lapply(.SD, as.numeric),.SDcols = xFEAT])

# fit model
model <- lgb.train(params = param, data = dt_trainDM, valid=dt_validDM)

# predict
pred <- predict(model, dt_testMT)

#Teste
result <- Metrics::rmse(pred, log1p(dt_test[["sales"]]))
print(result)
print(result / model$best_score)

#Gerando dados para plotagem
dt_test[, prediction := pred]
plot_data <- rbind(
  plot_data, dt_test[
    , 
    .(
      prediction = sum(expm1(prediction), na.rm = TRUE),
      actual = sum(sales, na.rm = TRUE),
      dept_id
    ), by = c("date")
  ]
)


#plot agregado
colors <- c("actual" = "darkred", "prediction" = "steelblue")
ggplot(data = plot_data, aes (x = date)) + 
  geom_point(aes (y = actual, color = "actual")) +
  geom_point(aes (y = prediction, color = "prediction")) +
  labs(y = "Actual X Prediction") + theme_classic() +
  scale_color_manual(values = colors) +
  facet_wrap(~ dept_id)

#diff Plot
plot_data <- data.table()
plot_data <- dt_test[, .(difference = round(expm1(prediction) - sales, 2), date), by = "item_id"]
ggplot(data = plot_data, aes (x = date)) +
  geom_point(aes (y = difference, group = item_id)) +
  labs(y = "Actual - Prediction") + theme_classic() +
  scale_y_continuous(limit = c(-10, 10), breaks = c(-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10))


#Importance
importance <- lgb.importance(model)
head(importance)
lgb.plot.importance(importance, top_n = 30L)

#Saving model
lgb.save(model, "Modelo_3000_2.txt")


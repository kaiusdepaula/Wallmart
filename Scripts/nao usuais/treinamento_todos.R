library(data.table)
library(lightgbm)
library(ggplot2)

evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  label_rolling = data.table(labels)
  label_rolling[, V1 := (V1 - shift(V1, 1)) ^ 2]
  err <- as.numeric(sqrt(
    (1 / 28) * (
      sum((labels - preds) ^ 2, na.rm = TRUE) /
        ((1 / 1913 - 1) * 
           sum(label_rolling$V1, na.rm = TRUE)
        ))
    )
  )
  return(list(name = "RMSSE", value = err, higher_better = FALSE))
}


#Lista de parâmetros de treinamento do modelo
param <- list(boosting_type = "gbdt",
              objective = "tweedie",
              tweedie_variance_power = 1.1,
              eval = evalerror,
              subsample = 0.5,
              subsample_freq = 1,
              learning_rate = 0.015,
              num_leaves = 2**8-1,
              min_data_in_leaf = 2**8-1,
              feature_fraction = 0.5,
              max_bin = 100,
              n_estimators = 500,
              boost_from_average = FALSE,
              verbose = -1,
              seed = 1995)

lojas <- c("CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", 
           "WI_2", "WI_3")

for (loja in lojas) {
  print(paste0("Treinando o modelo da loja ", loja))
  #Leitura dos dados
  dataset <- as.data.table(arrow::read_parquet(paste0("~/Wallmart/dados_de_treinamento/data_processada_", loja, ".parquet")))
  
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
    "date", "item_id", "snap_TX", "snap_WI", "sales", "store_id", "ordenador"
  ))
  
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
  stop()
  # predict
  dt_test[, predictions := predict(model, dt_testMT)]
  dt_test[, residuals_squared := ((sales - predictions) ^ 2)]

  #Teste RMSE
  # result <- dt_test[, Metrics::rmse(predictions, log1p(sales))]
  # print(result)
  # print(result / model$best_score)
  
  #Salvando o modelo
  
  lgb.save(model, paste0("Modelo_teste", loja, ".txt"))
  model <- lgb.load(paste0("Modelo_teste", loja, ".txt"))
  rm(list = setdiff(ls(), c("param", "lojas", "loja")))
  invisible(gc())
  
}






data <- data.table(model$record_evals$test$rmse$eval)
data[, id:= 1:.N]
ggplot(data = data, aes(y = as.numeric(V1), x = id)) + geom_line()

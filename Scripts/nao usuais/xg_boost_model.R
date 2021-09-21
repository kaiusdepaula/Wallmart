library(data.table)
library(xgboost)
library(ggplot2)


#Fun??o para metrica do modelo
evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- sqrt(mean((expm1(preds) - expm1(labels))^2, na.rm = TRUE))
  return(list(metric = "RMSE", value = err))
}


#Leitura dos dados
dataset <- as.data.table(arrow::read_parquet("~/Wallmart/Data/data_processada.parquet"))

#Declarando tipo
col_categoricas <- c(
  "weekday", "dept_id", "cat_id", 
  "item_id", "event_name_1", "event_name_2", "event_type_2",
  "event_type_1"
)

dataset[, (col_categoricas) := lapply(.SD, as.factor), .SDcols = col_categoricas]



#Defini??o dos valores a serem previstos
outcomes <- c("Target +1d", "Target +2d", "Target +3d", "Target +4d",
              "Target +5d", "Target +6d", "Target +7d")

#Colunas que v?o sair do escopo de treinamento 
xFEAT <- setdiff(names(dataset), c(
  "d", "wm_yr_wk", "date", "id", "item_id", "event_name_1", "event_type_1",
  "event_name_2", "event_type_2", "year", "snap_CA", "snap_TX", "snap_WI",
  "sales", "Target +1d", "Target +2d", 
  "Target +3d", "Target +4d", "Target +5d", "Target +6d", "Target +7d"
))

#Lista vazia para captar as m?tricas de valida??o
plot_data <- data.table()

#Lista de par?metros de treinamento do modelo
param <- list(booster = "gbtree",
              eta=0.4,
              gamma=0,
              base_score = 140,
              eval_metric = "mae")

for (outcome in outcomes) {
  dia_da_semana = fcase(outcome == "Target +1d", "Monday",
                        outcome == "Target +2d", "Tuesday",
                        outcome == "Target +3d", "Wednesday",
                        outcome == "Target +4d", "Thursday",
                        outcome == "Target +5d", "Friday",
                        outcome == "Target +6d", "Saturday",
                        outcome == "Target +7d", "Sunday")
  
  #Treinamento, valida??o e teste
  dataset_filtro <- dataset[weekday == "Sunday"]
  dt_train <- dataset_filtro[date < as.Date("2016-05-22") - 14L]
  dt_valid <- dataset_filtro[date >= as.Date("2016-05-22") - 14L & date < as.Date("2016-05-22") - 7L]
  dt_test <- dataset[date > as.Date("2016-05-22") - 7L]
  rm(dataset_filtro)
  
  #Treinamento
  dt_trainMT = as.matrix(dt_train[,lapply(.SD, as.numeric),.SDcols = xFEAT])
  dt_trainDM = xgb.DMatrix(data = dt_trainMT, label = dt_train[[outcome]])
  
  #Valida??o
  dt_validMT = as.matrix(dt_valid[,lapply(.SD, as.numeric),.SDcols = xFEAT])
  dt_validDM = xgb.DMatrix(data = dt_validMT, label = dt_valid[[outcome]])
  
  
  #Conjunto de teste
  dt_test <- dt_test[weekday == dia_da_semana]
  dt_testMT = as.matrix(dt_test[,lapply(.SD, as.numeric),.SDcols = xFEAT])
  
  # fit model
  print(paste0("Treinando o modelo da ", outcome))
  bst <- xgb.train(param = param, data = dt_trainDM, nrounds = 100,
                   watchlist <- list(train=dt_trainDM, valid=dt_validDM),
                   early_stopping_round = 3, maximize = FALSE)
  
  # predict
  pred <- predict(bst, dt_testMT)
  
  #Teste
  result <- Metrics::rmse(pred, dt_test[["sales"]])
  print(result)
  print(result / bst[["best_score"]])
  
  #Gerando dados para plotagem
  dt_test[, prediction := pred]
  plot_data <- rbind(
    plot_data, dt_test[
      , 
      .(
        prediction = sum(prediction, na.rm = TRUE),
        actual = sum(log1p(sales), na.rm = TRUE)
      ), by = c("date")
    ]
  )
  
}



#plot
colors <- c("actual" = "darkred", "prediction" = "steelblue")
ggplot(data = plot_data, aes (x = date)) + 
  geom_line(aes (y = actual, color = "actual")) +
  geom_line(aes (y = prediction, color = "prediction")) +
  labs(y = "Actual X Prediction", title = "Base score = 140") + theme_classic() +
  scale_color_manual(values = colors) 

#Importance matrix
importance <- xgb.importance(feature_names = bst$feature_names, model = bst)
head(importance)
xgb.plot.importance(importance)

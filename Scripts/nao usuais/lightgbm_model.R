library(data.table)
library(lightgbm)
library(ggplot2)

#Leitura dos dados
dataset <- as.data.table(arrow::read_parquet("~/Wallmart/Data/data_processada.parquet"))

#Declarando tipo
col_categoricas <- c(
  "weekday", "dept_id", "cat_id", 
  "item_id", "event_name_1", "event_name_2", "event_type_2",
  "event_type_1"
)
dataset[, (col_categoricas) := lapply(.SD, as.factor), .SDcols = col_categoricas]

#Definição dos valores a serem previstos
outcomes <- c("Target +1d", "Target +2d", "Target +3d", "Target +4d",
              "Target +5d", "Target +6d", "Target +7d")

#Colunas que vão sair do escopo de treinamento 
xFEAT <- setdiff(names(dataset), c(
  "d", "wm_yr_wk", "date", "id", "item_id", "event_name_1", "event_type_1",
  "event_name_2", "event_type_2", "year", "snap_CA", "snap_TX", "snap_WI",
  "sales", "Target +1d", "Target +2d", 
  "Target +3d", "Target +4d", "Target +5d", "Target +6d", "Target +7d"
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
              max_bin = N100,
              n_estimators = 50,
              boost_from_average = FALSE,
              verbose = -1,
              seed = 1995,
              early_stopping_rounds = 5)

for (outcome in outcomes) {
  dia_da_semana = fcase(outcome == "Target +1d", "Monday",
                        outcome == "Target +2d", "Tuesday",
                        outcome == "Target +3d", "Wednesday",
                        outcome == "Target +4d", "Thursday",
                        outcome == "Target +5d", "Friday",
                        outcome == "Target +6d", "Saturday",
                        outcome == "Target +7d", "Sunday")
  
  #Treinamento
  dt_train <- dataset[weekday == "Sunday"] #O último dia com observações
  dt_train <- dataset[date < as.Date("2016-05-22") - 14L]
  
  dt_trainMT = as.matrix(dt_train[,lapply(.SD, as.numeric),.SDcols = xFEAT])
  dt_trainDM = lgb.Dataset(dt_trainMT, label = dt_train[[outcome]])
  
  #Validação
  dt_valid <- dataset[date >= as.Date("2016-05-22") - 14L & date < as.Date("2016-05-22") - 7L]
  dt_valid <- dt_valid[weekday == dia_da_semana]
  
  dt_validMT = as.matrix(dt_valid[,lapply(.SD, as.numeric),.SDcols = xFEAT])
  dt_validDM = list(test = lgb.Dataset.create.valid(dt_trainDM, dt_validMT, label = log1p(dt_valid[["sales"]])))
  
  
  #Conjunto de teste
  dt_test <- dataset[date > as.Date("2016-05-22") - 7L]
  dt_test <- dt_test[weekday == dia_da_semana]
  
  dt_testMT = as.matrix(dt_test[,lapply(.SD, as.numeric),.SDcols = xFEAT])
  
  # fit model
  print(paste0("Treinando o modelo da ", outcome))
  model <- lgb.train(params = param, data = dt_trainDM, valid=dt_validDM)
  
  # predict
  pred <- predict(model, dt_testMT)
  
  #Teste
  result <- Metrics::rmse(pred, log1p(dt_test[["sales"]]))
  print(result)
  print(result / model$best_score)
  
  #Gerando dados para plotagem
  dt_test[, prediction := pred]
  dt_test[id == "FOODS_3_586_CA_2_evaluation"]
  plot_data <- rbind(
    plot_data, dt_test[
      , 
      .(
        prediction = sum(expm1(prediction), na.rm = TRUE),
        actual = sum(sales, na.rm = TRUE)
      ), by = c("date")
    ]
  )
  
}



#plot
colors <- c("actual" = "darkred", "prediction" = "steelblue")
ggplot(data = plot_data, aes (x = date)) + 
  geom_line(aes (y = actual, color = "actual")) +
  geom_line(aes (y = prediction, color = "prediction")) +
  labs(y = "Actual X Prediction") + theme_classic() +
  scale_color_manual(values = colors) 

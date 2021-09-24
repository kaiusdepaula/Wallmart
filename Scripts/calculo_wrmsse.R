############################################
#Cálculo WRMSSE
############################################
library(data.table)
library(lightgbm)
lojas <- c("CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", 
           "WI_2", "WI_3")

dt_train <- data.table()
dt_valid <- data.table()
for (loja in lojas) {
  #Leitura dos dados
  dataset <- as.data.table(arrow::read_parquet(paste0("~/projetos/Wallmart/dados_de_treinamento/data_processada_", loja, ".parquet")))
  
  cols <- c("dept_id", "cat_id", "sales", "sell_price", "item_id", 
            "store_id", "state_id", "dia")
  
  #Filtrando o periodo usado para treinar
  temp <- dataset[dia < 1914]
  temp <- temp[, ..cols]
  
  #Filtrando o periodo usado para prever
  temp2 <- dataset[dia >= 1914 & dia <= 1941]
  
  #Declarando tipo
  col_categoricas <- c(
    "store_id", "dept_id", "cat_id",
    "item_id", "event_name_1", "event_name_2", "event_type_2",
    "event_type_1", "snap_CA", "month", "weekday", 
    "yearday", "year"
  )
  temp2[, (col_categoricas) := lapply(.SD, as.factor), .SDcols = col_categoricas]
  
  #Colunas que vão sair do escopo de previsão
  xFEAT <- setdiff(names(dataset), c(
    "date", "sales", "ordenador", "state_id"
  ))
  
  #Matriz de preditores
  dt_predMT = as.matrix(temp2[,lapply(.SD, as.numeric),.SDcols = xFEAT])
  
  #Carregando o modelo
  model <- lgb.load(paste0("Modelo_teste_cv", loja, ".txt"))
  temp2[, preds := expm1(predict(model, dt_predMT))]
  temp2[, revenue := sales * sell_price]
  
  cols <- c("dept_id", "cat_id", "sales", "sell_price", "item_id", 
            "store_id", "state_id", "dia", "preds", "revenue")
  temp2 <- temp2[, ..cols]
  
  dt_train <- rbind(dt_train, temp)
  dt_valid <- rbind(dt_valid, temp2)
  
  rm(dataset, temp, temp2, cols, model, dt_predMT)
  invisible(gc())
}

#Cálculo do WRMSSE

#Nível 1
#Retorno de 1 valor por dia
data <- dt_train[, .(sales, dia)] 
data <- data[
  , 
  .(
    sales = sum(sales, na.rm = TRUE)
  ), by = dia
]

data2 <- dt_valid[, .(sales, preds, dia)]
data2 <- data2[
  , 
  .(
    sales = sum(sales, na.rm = TRUE), 
    preds = sum(preds, na.rm = TRUE)
  ), by = dia
]

RMSE_future <- ModelMetrics::rmse(data2$sales, data2$preds) #Numerador
RMSE_historical <- data[, sqrt(mean((diff(sales) ** 2), na.rm = TRUE))] #Denominador
RMSSE <- RMSE_future / RMSE_historical

#weights
weights <- 1 #Unica serie de tempo da agregação

WRMSSE_1 <- ((1/12) * RMSSE * weights)

#Nível 2
#Retorno de 3 valores por dia
data <- dt_train[, .(sales, dia, state_id)] 
data <- data[
  , 
  .(
    sales = sum(sales, na.rm = TRUE)
  ), by = c("dia", "state_id")
]

data2 <- dt_valid[, .(sales, preds, dia, state_id)]
data2 <- data2[
  , 
  .(
    sales = sum(sales, na.rm = TRUE), 
    preds = sum(preds, na.rm = TRUE)
  ), by = c("dia", "state_id")
]

#adicionar loop
RMSE_future <- ModelMetrics::rmse(data2$sales, data2$preds) #Numerador
RMSE_historical <- data[, sqrt(mean((diff(sales) ** 2), na.rm = TRUE))] #Denominador
RMSSE <- RMSE_future / RMSE_historical

#weights
data3 <- dt_valid[
  , 
  .(
    revenue = sum(revenue, na.rm = TRUE)
  ), by = dia
]

weights <- data3[, sum(revenue) / sum(revenue)]

WRMSSE_1 <- ((1/12) * RMSSE * weights)
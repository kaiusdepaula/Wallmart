library(data.table)
library(lightgbm)
       
lojas <- c("CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", 
           "WI_2", "WI_3")

setwd("~/projetos/Wallmart/modelos")
#Prevendo public scoreboard

for (loja in lojas) {
  #Leitura dos dados
  dataset <- as.data.table(arrow::read_parquet(paste0("~/projetos/Wallmart/dados_de_treinamento/data_processada_", loja, ".parquet")))
  
  #Filtrando o periodo usado para prever
  dataset <- dataset[dia >= 1914 & dia <= 1941]
  
  #Declarando tipo
  col_categoricas <- c(
    "store_id", "dept_id", "cat_id",
    "item_id", "event_name_1", "event_name_2", "event_type_2",
    "event_type_1", "snap_CA", "month", "weekday", 
    "yearday", "year"
  )
  dataset[, (col_categoricas) := lapply(.SD, as.factor), .SDcols = col_categoricas]
  
  #Colunas que vão sair do escopo de previsão
  xFEAT <- setdiff(names(dataset), c(
    "date", "sales", "ordenador", "state_id"
  ))
  
  #Matriz de preditores
  dt_predMT = as.matrix(dataset[,lapply(.SD, as.numeric),.SDcols = xFEAT])
  
  #Carregando o modelo
  model <- lgb.load(paste0("Modelo_teste_cv", loja, ".txt"))
  
  #Gerando mecanismo que salva as informações
  chaves_sales <- c("date", "dia", "item_id", "store_id", "ordenador", "dept_id", "state_id", "cat_id")
  pred <- dataset[, ..chaves_sales]
  
  #Gerando previsões
  temp <- predict(model, dt_predMT)
  
  #Salvando previsões no datatable pred
  pred[, predicao := temp]
  
  #Salvando previsões
  arrow::write_parquet(pred, paste0("~/projetos/Wallmart/previsao/data_prevista_valid_", loja, ".parquet"))
  
  rm(list = setdiff(ls(), c("lojas")))
}


#Prevendo private scoreboard

for (loja in lojas) {
  #Leitura dos dados
  dataset <- as.data.table(arrow::read_parquet(paste0("~/projetos/Wallmart/dados_de_treinamento/data_processada_", loja, ".parquet")))
  
  #Filtrando o periodo usado para prever
  dataset <- dataset[dia >= 1942]
  
  #Declarando tipo
  col_categoricas <- c(
    "store_id", "dept_id", "cat_id",
    "item_id", "event_name_1", "event_name_2", "event_type_2",
    "event_type_1", "snap_CA", "month", "weekday", 
    "yearday", "year"
  )
  dataset[, (col_categoricas) := lapply(.SD, as.factor), .SDcols = col_categoricas]
  
  #Colunas que vão sair do escopo de previsão
  xFEAT <- setdiff(names(dataset), c(
    "date", "sales", "ordenador", "state_id"
  ))
  
  #Matriz de preditores
  dt_predMT = as.matrix(dataset[,lapply(.SD, as.numeric),.SDcols = xFEAT])
  
  #Carregando o modelo
  model <- lgb.load(paste0("Modelo_teste_cv", loja, ".txt"))
  
  #Gerando mecanismo que salva as informações
  chaves_sales <- c("date", "dia", "item_id", "store_id", "ordenador", "dept_id", "state_id", "cat_id")
  pred <- dataset[, ..chaves_sales]
  
  #Gerando previsões
  temp <- predict(model, dt_predMT)
  
  #Salvando previsões no datatable pred
  pred[, predicao := temp]
  
  #Salvando previsões
  arrow::write_parquet(pred, paste0("~/projetos/Wallmart/previsao/data_prevista_eval_", loja, ".parquet"))
  
  rm(list = setdiff(ls(), c("lojas", "loja", "empilhado")))
}


submission <- data.table()

#Gerando arquivos empilhados valid

empilhado <- data.table()

for (loja in lojas) {
  pred <- as.data.table(arrow::read_parquet(paste0("~/projetos/Wallmart/previsao/data_prevista_valid_", loja, ".parquet")))
  empilhado <- rbind(empilhado, pred)
  rm(pred)
}


empilhado[, predicao := expm1(predicao)]
empilhado[, item_id := paste0(item_id, "_", store_id, "_validation")]
empilhado[, day := fcase(
  date == "2016-04-25", "F1",
  date == "2016-04-26", "F2",
  date == "2016-04-27", "F3",
  date == "2016-04-28", "F4",
  date == "2016-04-29", "F5",
  date == "2016-04-30", "F6",
  date == "2016-05-01", "F7",
  date == "2016-05-02", "F8",
  date == "2016-05-03", "F9",
  date == "2016-05-04", "F10",
  date == "2016-05-05", "F11",
  date == "2016-05-06", "F12",
  date == "2016-05-07", "F13",
  date == "2016-05-08", "F14",
  date == "2016-05-09", "F15",
  date == "2016-05-10", "F16",
  date == "2016-05-11", "F17",
  date == "2016-05-12", "F18",
  date == "2016-05-13", "F19",
  date == "2016-05-14", "F20",
  date == "2016-05-15", "F21",
  date == "2016-05-16", "F22",
  date == "2016-05-17", "F23",
  date == "2016-05-18", "F24",
  date == "2016-05-19", "F25",
  date == "2016-05-20", "F26",
  date == "2016-05-21", "F27",
  date == "2016-05-22", "F28"
)]

setnames(empilhado, "item_id", "id")

dataset <- dcast(empilhado, id + ordenador ~ day, value.var = "predicao")

setcolorder(
  dataset, c(
    "id", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", 
    "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", 
    "F20", "F21", "F22", "F23", "F24", "F25", "F26", "F27", "F28"
  )
)

dataset <- dataset[order(ordenador)]
dataset[, ordenador := NULL]

submission <- rbind(submission, dataset)



#Gerando arquivos empilhados evaluation

empilhado <- data.table()

for (loja in lojas) {
  pred <- as.data.table(arrow::read_parquet(paste0("~/projetos/Wallmart/previsao/data_prevista_eval_", loja, ".parquet")))
  empilhado <- rbind(empilhado, pred)
  rm(pred)
}

empilhado[, predicao := expm1(predicao)]
empilhado[, item_id := paste0(item_id, "_", store_id, "_evaluation")]
empilhado[, day := fcase(
  date == "2016-05-23", "F1",
  date == "2016-05-24", "F2",
  date == "2016-05-25", "F3",
  date == "2016-05-26", "F4",
  date == "2016-05-27", "F5",
  date == "2016-05-28", "F6",
  date == "2016-05-29", "F7",
  date == "2016-05-30", "F8",
  date == "2016-05-31", "F9",
  date == "2016-06-01", "F10",
  date == "2016-06-02", "F11",
  date == "2016-06-03", "F12",
  date == "2016-06-04", "F13",
  date == "2016-06-05", "F14",
  date == "2016-06-06", "F15",
  date == "2016-06-07", "F16",
  date == "2016-06-08", "F17",
  date == "2016-06-09", "F18",
  date == "2016-06-10", "F19",
  date == "2016-06-11", "F20",
  date == "2016-06-12", "F21",
  date == "2016-06-13", "F22",
  date == "2016-06-14", "F23",
  date == "2016-06-15", "F24",
  date == "2016-06-16", "F25",
  date == "2016-06-17", "F26",
  date == "2016-06-18", "F27",
  date == "2016-06-19", "F28"
)]

setnames(empilhado, "item_id", "id")

dataset <- dcast(empilhado, id + ordenador ~ day, value.var = "predicao")

setcolorder(
  dataset, c(
    "id", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", 
    "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", 
    "F20", "F21", "F22", "F23", "F24", "F25", "F26", "F27", "F28"
  )
)

dataset <- dataset[order(ordenador)]
dataset[, ordenador := NULL]

submission <- rbind(submission, dataset)

fwrite(submission, file = "~/projetos/Wallmart/submission.csv")

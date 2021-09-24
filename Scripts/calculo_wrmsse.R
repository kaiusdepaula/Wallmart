############################################
#Cálculo WRMSSE
############################################
library(data.table)
library(lightgbm)
lojas <- c("CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", 
           "WI_2", "WI_3")

setwd("~/projetos/Wallmart/modelos")
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
#Retorno de 1 valor por dia - Total
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
#Retorno de 3 valores por dia - State
RMSSE <- c()
for (state in unique(dt_valid$state_id)) {
  data <- dt_train[, .(sales, dia, state_id)]
  data <- data[state_id == state]
  data <- data[
    , 
    .(
      sales = sum(sales, na.rm = TRUE)
    ), by = c("dia")
  ]
  
  data2 <- dt_valid[, .(sales, preds, dia, state_id)]
  data2 <- data2[state_id == state]
  data2 <- data2[
    , 
    .(
      sales = sum(sales, na.rm = TRUE), 
      preds = sum(preds, na.rm = TRUE)
    ), by = c("dia")
  ]
  RMSE_future <- ModelMetrics::rmse(data2$sales, data2$preds) #Numerador
  RMSE_historical <- data[, sqrt(mean((diff(sales) ** 2), na.rm = TRUE))] #Denominador
  RMSSE <- rbind(RMSSE, RMSE_future / RMSE_historical)
}

#weights
data3 <- dt_valid[
  , 
  .(
    revenue = sum(revenue, na.rm = TRUE)
  ), by = state_id
]

weights <- data3[, revenue / sum(revenue)]

WRMSSE_2 <- sum((1/12) * RMSSE * weights)

#Nível 3
#Retorno de 10 valores por dia - Store
RMSSE <- c()
for (store in unique(dt_valid$store_id)) {
  data <- dt_train[, .(sales, dia, store_id)]
  data <- data[store_id == store]
  data <- data[
    , 
    .(
      sales = sum(sales, na.rm = TRUE)
    ), by = c("dia")
  ]
  
  data2 <- dt_valid[, .(sales, preds, dia, store_id)]
  data2 <- data2[store_id == store]
  data2 <- data2[
    , 
    .(
      sales = sum(sales, na.rm = TRUE), 
      preds = sum(preds, na.rm = TRUE)
    ), by = c("dia")
  ]
  RMSE_future <- ModelMetrics::rmse(data2$sales, data2$preds) #Numerador
  RMSE_historical <- data[, sqrt(mean((diff(sales) ** 2), na.rm = TRUE))] #Denominador
  RMSSE <- rbind(RMSSE, RMSE_future / RMSE_historical)
}

#weights
data3 <- dt_valid[
  , 
  .(
    revenue = sum(revenue, na.rm = TRUE)
  ), by = store_id
]

weights <- data3[, revenue / sum(revenue)]

WRMSSE_3 <- sum((1/12) * RMSSE * weights)

#Nível 4
#Retorno de 3 valores por dia - Category
RMSSE <- c()
for (cat in unique(dt_valid$cat_id)) {
  data <- dt_train[, .(sales, dia, cat_id)]
  data <- data[cat_id == cat]
  data <- data[
    , 
    .(
      sales = sum(sales, na.rm = TRUE)
    ), by = c("dia")
  ]
  
  data2 <- dt_valid[, .(sales, preds, dia, cat_id)]
  data2 <- data2[cat_id == cat]
  data2 <- data2[
    , 
    .(
      sales = sum(sales, na.rm = TRUE), 
      preds = sum(preds, na.rm = TRUE)
    ), by = c("dia")
  ]
  RMSE_future <- ModelMetrics::rmse(data2$sales, data2$preds) #Numerador
  RMSE_historical <- data[, sqrt(mean((diff(sales) ** 2), na.rm = TRUE))] #Denominador
  RMSSE <- rbind(RMSSE, RMSE_future / RMSE_historical)
}

#weights
data3 <- dt_valid[
  , 
  .(
    revenue = sum(revenue, na.rm = TRUE)
  ), by = cat_id
]

weights <- data3[, revenue / sum(revenue)]

WRMSSE_4 <- sum((1/12) * RMSSE * weights)

#Nível 5
#Retorno de 7 valores por dia - Department
RMSSE <- c()
for (dept in unique(dt_valid$dept_id)) {
  data <- dt_train[, .(sales, dia, dept_id)]
  data <- data[dept_id == dept]
  data <- data[
    , 
    .(
      sales = sum(sales, na.rm = TRUE)
    ), by = c("dia")
  ]
  
  data2 <- dt_valid[, .(sales, preds, dia, dept_id)]
  data2 <- data2[dept_id == dept]
  data2 <- data2[
    , 
    .(
      sales = sum(sales, na.rm = TRUE), 
      preds = sum(preds, na.rm = TRUE)
    ), by = c("dia")
  ]
  RMSE_future <- ModelMetrics::rmse(data2$sales, data2$preds) #Numerador
  RMSE_historical <- data[, sqrt(mean((diff(sales) ** 2), na.rm = TRUE))] #Denominador
  RMSSE <- rbind(RMSSE, RMSE_future / RMSE_historical)
}

#weights
data3 <- dt_valid[
  , 
  .(
    revenue = sum(revenue, na.rm = TRUE)
  ), by = dept_id
]

weights <- data3[, revenue / sum(revenue)]

WRMSSE_5 <- sum((1/12) * RMSSE * weights)

#Nível 6
#Retorno de 9 valores por dia - State, category
RMSSE <- c()
for (state in unique(dt_valid$state_id)) {
  for (cat in unique(dt_valid$cat_id)) {
    data <- dt_train[, .(sales, dia, state_id, cat_id)]
    data <- data[state_id == state & cat_id == cat]
    data <- data[
      , 
      .(
        sales = sum(sales, na.rm = TRUE)
      ), by = c("dia")
    ]
    
    data2 <- dt_valid[, .(sales, preds, dia, state_id, cat_id)]
    data2 <- data2[state_id == state & cat_id == cat]
    data2 <- data2[
      , 
      .(
        sales = sum(sales, na.rm = TRUE), 
        preds = sum(preds, na.rm = TRUE)
      ), by = c("dia")
    ]
    RMSE_future <- ModelMetrics::rmse(data2$sales, data2$preds) #Numerador
    RMSE_historical <- data[, sqrt(mean((diff(sales) ** 2), na.rm = TRUE))] #Denominador
    RMSSE <- rbind(RMSSE, RMSE_future / RMSE_historical)
  }
}

#weights
data3 <- dt_valid[
  , 
  .(
    revenue = sum(revenue, na.rm = TRUE)
  ), by = c("state_id", "cat_id")
]

weights <- data3[, revenue / sum(revenue)]

WRMSSE_6 <- sum((1/12) * RMSSE * weights)

#Nível 7
#Retorno de 21 valores por dia - State, department
RMSSE <- c()
for (state in unique(dt_valid$state_id)) {
  for (dept in unique(dt_valid$dept_id)) {
    data <- dt_train[, .(sales, dia, state_id, dept_id)]
    data <- data[state_id == state & dept_id == dept]
    data <- data[
      , 
      .(
        sales = sum(sales, na.rm = TRUE)
      ), by = c("dia")
    ]
    
    data2 <- dt_valid[, .(sales, preds, dia, state_id, dept_id)]
    data2 <- data2[state_id == state & dept_id == dept]
    data2 <- data2[
      , 
      .(
        sales = sum(sales, na.rm = TRUE), 
        preds = sum(preds, na.rm = TRUE)
      ), by = c("dia")
    ]
    RMSE_future <- ModelMetrics::rmse(data2$sales, data2$preds) #Numerador
    RMSE_historical <- data[, sqrt(mean((diff(sales) ** 2), na.rm = TRUE))] #Denominador
    RMSSE <- rbind(RMSSE, RMSE_future / RMSE_historical)
  }
}

#weights
data3 <- dt_valid[
  , 
  .(
    revenue = sum(revenue, na.rm = TRUE)
  ), by = c("state_id", "dept_id")
]

weights <- data3[, revenue / sum(revenue)]

WRMSSE_7 <- sum((1/12) * RMSSE * weights)

#Nível 8
#Retorno de 30 valores por dia - Store, category
RMSSE <- c()
for (store in unique(dt_valid$store_id)) {
  for (cat in unique(dt_valid$cat_id)) {
    data <- dt_train[, .(sales, dia, store_id, cat_id)]
    data <- data[store_id == store & cat_id == cat]
    data <- data[
      , 
      .(
        sales = sum(sales, na.rm = TRUE)
      ), by = c("dia")
    ]
    
    data2 <- dt_valid[, .(sales, preds, dia, store_id, cat_id)]
    data2 <- data2[store_id == store & cat_id == cat]
    data2 <- data2[
      , 
      .(
        sales = sum(sales, na.rm = TRUE), 
        preds = sum(preds, na.rm = TRUE)
      ), by = c("dia")
    ]
    RMSE_future <- ModelMetrics::rmse(data2$sales, data2$preds) #Numerador
    RMSE_historical <- data[, sqrt(mean((diff(sales) ** 2), na.rm = TRUE))] #Denominador
    RMSSE <- rbind(RMSSE, RMSE_future / RMSE_historical)
  }
}

#weights
data3 <- dt_valid[
  , 
  .(
    revenue = sum(revenue, na.rm = TRUE)
  ), by = c("store_id", "cat_id")
]

weights <- data3[, revenue / sum(revenue)]

WRMSSE_8 <- sum((1/12) * RMSSE * weights)

#Nível 9
#Retorno de 70 valores por dia - Store, department
RMSSE <- c()
for (store in unique(dt_valid$store_id)) {
  for (dept in unique(dt_valid$dept_id)) {
    data <- dt_train[, .(sales, dia, store_id, dept_id)]
    data <- data[store_id == store & dept_id == dept]
    data <- data[
      , 
      .(
        sales = sum(sales, na.rm = TRUE)
      ), by = c("dia")
    ]
    
    data2 <- dt_valid[, .(sales, preds, dia, store_id, dept_id)]
    data2 <- data2[store_id == store & dept_id == dept]
    data2 <- data2[
      , 
      .(
        sales = sum(sales, na.rm = TRUE), 
        preds = sum(preds, na.rm = TRUE)
      ), by = c("dia")
    ]
    RMSE_future <- ModelMetrics::rmse(data2$sales, data2$preds) #Numerador
    RMSE_historical <- data[, sqrt(mean((diff(sales) ** 2), na.rm = TRUE))] #Denominador
    RMSSE <- rbind(RMSSE, RMSE_future / RMSE_historical)
  }
}

#weights
data3 <- dt_valid[
  , 
  .(
    revenue = sum(revenue, na.rm = TRUE)
  ), by = c("store_id", "dept_id")
]

weights <- data3[, revenue / sum(revenue)]

WRMSSE_9 <- sum((1/12) * RMSSE * weights)

######################### A partir daqui se torna inviável sem otimizar

#Nível 10
#Retorno de 3049 valores por dia - item_id
RMSSE <- c()
for (item in unique(dt_valid$item_id)) {
  data <- dt_train[, .(sales, dia, item_id)]
  data <- data[item_id == item]
  
  data2 <- dt_valid[, .(sales, preds, dia, item_id)]
  data2 <- data2[item_id == item]

  RMSE_future <- ModelMetrics::rmse(data2$sales, data2$preds) #Numerador
  RMSE_historical <- data[, sqrt(mean((diff(sales) ** 2), na.rm = TRUE))] #Denominador
  RMSSE <- rbind(RMSSE, RMSE_future / RMSE_historical)
}

#weights
data3 <- dt_valid[
  , 
  .(
    revenue = sum(revenue, na.rm = TRUE)
  ), by = c("item_id")
]

weights <- data3[, revenue / sum(revenue)]

WRMSSE_10 <- sum((1/12) * RMSSE * weights)

#Nível 11
#Retorno de 9147 valores por dia - item_id, state
RMSSE <- c()
for (state in unique(dt_valid$state_id)) {
  for (item in unique(dt_valid$item_id)) {
    data <- dt_train[, .(sales, dia, item_id, state_id)]
    data <- data[item_id == item & state_id == state]
    
    data2 <- dt_valid[, .(sales, preds, dia, item_id, state_id)]
    data2 <- data2[item_id == item & state_id == state]
    
    RMSE_future <- ModelMetrics::rmse(data2$sales, data2$preds) #Numerador
    RMSE_historical <- data[, sqrt(mean((diff(sales) ** 2), na.rm = TRUE))] #Denominador
    RMSSE <- rbind(RMSSE, RMSE_future / RMSE_historical)
  }
}


#weights
data3 <- dt_valid[
  , 
  .(
    revenue = sum(revenue, na.rm = TRUE)
  ), by = c("item_id", "state_id")
]

weights <- data3[, revenue / sum(revenue)]

WRMSSE_11 <- sum((1/12) * RMSSE * weights)

#Nível 12
#Retorno de 30490 valores por dia - item_id, store
RMSSE <- c()
for (store in unique(dt_valid$store_id)) {
  for (item in unique(dt_valid$item_id)) {
    data <- dt_train[, .(sales, dia, item_id, store_id)]
    data <- data[item_id == item & store_id == store]
    
    data2 <- dt_valid[, .(sales, preds, dia, item_id, store_id)]
    data2 <- data2[item_id == item & store_id == store]
    
    RMSE_future <- ModelMetrics::rmse(data2$sales, data2$preds) #Numerador
    RMSE_historical <- data[, sqrt(mean((diff(sales) ** 2), na.rm = TRUE))] #Denominador
    RMSSE <- rbind(RMSSE, RMSE_future / RMSE_historical)
  }
}


#weights
data3 <- dt_valid[
  , 
  .(
    revenue = sum(revenue, na.rm = TRUE)
  ), by = c("item_id", "store_id")
]

weights <- data3[, revenue / sum(revenue)]

WRMSSE_12 <- sum((1/12) * RMSSE * weights)

WRMSSE <- (WRMSSE_1 + WRMSSE_2 + WRMSSE_3 + WRMSSE_4 + WRMSSE_5 + WRMSSE_6 +
             WRMSSE_7 + WRMSSE_8 + WRMSSE_9 + WRMSSE_10 + WRMSSE_11 + WRMSSE_12)

library(data.table)

#Trazendo os dados
sales <- fread("~/projetos/Wallmart/Data/sales_train_evaluation.csv")
#Marcador de ordem
sales[, ordenador := 1L:.N]
sell <- fread("~/projetos/Wallmart/Data/sell_prices.csv")
calendar <- fread("~/projetos/Wallmart/Data/calendar.csv")
calendar_sem_extra <- names(calendar)[c(1, 2, 7:14)]
calendar = calendar[, ..calendar_sem_extra]
rm(calendar_sem_extra)

#Chaves relevantes
chaves_sales <- c('id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'ordenador')

#Adicionando 28 dias a mais em sales
days_add <- sales[, ..chaves_sales]
for (i in 1942L:1969) {
  days_add[, paste0("d_", i) := NA_integer_]
}
sales <- merge(days_add, sales, chaves_sales)

#Em meio a conservar memória, vou salvar cada departamento em um arquivo diferente
#e treiná-los separadamente

lojas <- c("CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", 
           "WI_2", "WI_3")

for (loja in lojas) {
  
  #Filtrando para ter apenas uma loja em sales
  sales_filtrado <- sales[store_id == loja]
  
  #Filtrando para ter apenas uma loja em sell
  sell_filtrado <- sell[store_id == loja]
  invisible(gc())
  
  #Mudando a apresentação dos dados de sales
  dataset <- melt.data.table(sales_filtrado,
                             id.vars = chaves_sales,
                             variable.name = 'd',
                             value.name = 'sales')
  rm(sales_filtrado)
  #dataset[, store_id := NULL]
  invisible(gc())
  
  #Juntando os dados
  
  dataset <- merge(
    dataset,
    calendar,
    on = "d"
  )
  invisible(gc())
  
  dataset <- merge(
    dataset,
    sell_filtrado,
    by.x = c("item_id", "wm_yr_wk", "store_id"),
    by.y = c("item_id", "wm_yr_wk", "store_id")
  )
  
  rm(sell_filtrado)
  invisible(gc())
  
  
  #Mudando o tipo da variável date
  dataset[, date := as.Date(date, format = "%Y%m%d")]
  
  #Criando a variáveis de tempo
  dataset[, month := month(date)]
  dataset[, weekday := wday(date)]
  dataset[, yearday := yday(date)]
  dataset[, year := year(date)]
  
  #Transformando variáveis de tempo em ciclos de sen\cos
  #Não usarei pois afeta a performace dos modelos de árvore
  # dataset[
  #   ,
  #   `:=` (
  #     month_sin = sin(2 * pi * month / 12),
  #     month_cos = cos(2 * pi * month / 12),
  #     weekday_sin = sin(2 * pi * weekday / 7),
  #     weekday_cos = cos(2 * pi * weekday / 7),    
  #     yearday_sin = sin(2 * pi * yearday / 365),
  #     yearday_cos = cos(2 * pi * yearday / 365)
  #   )
  # ]
  
  #Pegando dia para facilitar na validação
  dataset[, dia := as.numeric(date - min(date) + 1)]
  
  #Modificando a apresentação do dataset
  
  setcolorder(dataset, c("date", "id", "dept_id", "cat_id", "sales", "sell_price"))
  
  #Reordenando por id e data
  dataset <- dataset[order(item_id, date, dept_id)]
  
  #Normalizando dados de sell_price
  dataset[, sell_price_max := max(sell_price, na.rm = TRUE), by = c("item_id")]
  dataset[, sell_price_min := min(sell_price, na.rm = TRUE), by = c("item_id")]
  dataset[, sell_price_std := sd(sell_price, na.rm = TRUE), by = c("item_id")]
  dataset[, sell_price_mean := mean(sell_price, na.rm = TRUE), by = c("item_id")]
  dataset[, sell_price_norm := (sell_price - sell_price_min) / (sell_price_max - sell_price_min)]
  
  #Criando price momentum inspirado no notebook vencedor do kaggle
  dataset[, price_momentum := (sell_price / shift(sell_price, 7)), by = c("item_id")]
  dataset[, price_momentum_m := (sell_price / mean(sell_price, na.rm = TRUE)), by = c("item_id", "month")]
  dataset[, price_momentum_y := (sell_price / mean(sell_price, na.rm = TRUE)), by = c("item_id", "year")]
  
  #Criando features defasadas
  dataset <- dataset[order(item_id, dia)]
  qtd_lag <- c(1, 2, 3, 5, 7, 14, 28)
  for (i in qtd_lag) {
    dataset[, paste0("Venda_defasada_+", i, "d") := shift(log1p(sales), i), by = c("item_id")]
  }
  
  #Criando features de média em cima de features defasadas
  
  #Média movel label defasada
  dataset[, media_movel_venda := frollmean(`Venda_defasada_+28d`, 3), by = c("item_id")]
  
  
  #Agregações por categoria (defasagem 14 e 28 dias)
  defasagens <- c(7, 14, 28)
  for (defasagem in defasagens) {
    dataset[, paste0("venda_media_total_dia_", defasagem) := mean(
      get(paste0("Venda_defasada_+", defasagem, "d")), na.rm = TRUE
    ), by = c("dia")] 
    
    dataset[, paste0("venda_media_total_dia_item_", defasagem) := mean(
      get(paste0("Venda_defasada_+", defasagem, "d")), na.rm = TRUE
    ), by = c("dia", "item_id")] 
    
    dataset[, paste0("venda_media_dia_dept_", defasagem) := mean(
      get(paste0("Venda_defasada_+", defasagem, "d")), na.rm = TRUE
    ), by = c("dia", "dept_id")]
    
    dataset[, paste0("venda_media_dia_cat_", defasagem) := mean(
      get(paste0("Venda_defasada_+", defasagem, "d")), na.rm = TRUE
    ), by = c("dia", "cat_id")]
  }
  
  #Medias movel preco
  qtd_dias_media_movel_preco <- c(30)
  for (i in qtd_dias_media_movel_preco) {
    dataset[, paste0("media_movel_", qtd_dias_media_movel_preco, "_dias_preco") := frollmean(sell_price, qtd_dias_media_movel_preco),
            by = c("item_id")]
  }
  
  dataset[, c("wm_yr_wk", "id", "d") := NULL]
  
  
  arrow::write_parquet(dataset, paste0("~/projetos/Wallmart/dados_de_treinamento/data_processada_", loja, ".parquet"))
  rm(dataset)
  invisible(gc())
}

################################### RASCUNHO #################################################

#Feature de variacao percentual de preco
#qtd_dias_de_diferenca <- c(7, 14, 21)
#for (i in qtd_dias_de_diferenca) {
#   dataset[, paste0("variacao_percentual_preco", i, "_dias") := ((shift(sell_price, i) - sell_price) / sell_price),
#           by = c("item_id")]
# }

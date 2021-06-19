################################################################################
###########----------------- Machine Learning  ------------------###############
################################################################################

# Capacitador: André Omar Chávez Panduro
# email: andrecp38@gmail.com
# Tema: Aprendizaje SemiSupervisado
############################################################################

# Para limpiar el workspace, por si hubiera algun dataset 
# o informacion cargada
rm(list = ls())

# Cambiar el directorio de trabajo
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()


############
# Paquetes #
############

library(arules)
library(arulesViz)
library(RColorBrewer)
library(dplyr)
library(ggplot2)

#########################
# EJEMPLO INTRODUCTORIO #
#########################

# Paso 1: Cargando los datos

# Cargar la data en un matrix sparse
# Tenemos en total 4 trx
library(arules)
datos.a <- read.transactions("Ejemplo apriori.csv", sep = ",")

summary(datos.a)

# Conviertiendolo en un data frame
datos.df <- as(datos.a, Class = "data.frame")

# Paso 2: Explorando la data

inspect(datos.a[1])
inspect(datos.a[3])

# Mostrar todas las transacciones
inspect(datos.a)

# Examinar la frecuencia de items
itemFrequency(datos.a)

# Graficar la frecuencia de items
# Los items o objetos que se han consumido mas!
itemFrequencyPlot(datos.a, 
                  support = 0.5)

itemFrequencyPlot(datos.a, 
                  support = 0.5,
                  type ="absolute")

itemFrequencyPlot(datos.a, 
                  topN = 4,
                  col=brewer.pal(8,'Pastel2'),
                  main='Relative Item Frequency Plot',
                  ylab="Item Frequency (Relative)")

# Visualizar la matriz sparse 
# Aplicacion con nuestra real!
image(datos.a)

# Paso 3: Encontrar las reglas de asociacion
library(arules)

# Por defecto asume:
#     support = 0.1, 
#     confianza m?nima = 0.8, 
#     minimo de items (minlen) =  1
#     maximo de tems (maxlen) = 10


# Reglas de asociacion con los parametros por defecto

apriori(datos.a)

# Configurando los niveles de soporte y confianza 

datosrules <- apriori(datos.a, 
                        parameter = list(support = 0.5, 
                                         confidence = 0.001, 
                                         minlen = 2, 
                                         maxlen=10))
datosrules

# Paso 4: Evaluando las reglas de asociacion

# Resumiendo las reglas de asociacion
summary(datosrules)

# Mostrar la primera regla
inspect(datosrules[1])

# Mostrar todas las reglas
inspect(datosrules)

# Paso 5: Mejorando las reglas de asociacion

# Ordenando las reglas por el lift
inspect(sort(datosrules, by = "lift"))

# Ordenando las reglas por el soporte
inspect(sort(datosrules, by = "support"))

# Graficando el soporte y la confianza de las reglas
library(arulesViz)
plot(datosrules, method = "scatterplot")

# Representacion Grafica de las Reglas
plot(datosrules,
     method = "graph",
     control = list(type = "items"))

# Encontrando reglas redundantes
inspect(datosrules[is.redundant(datosrules)])

# Encontrando reglas no redundantes
inspect(datosrules[!is.redundant(datosrules)])

# Eliminando reglas redundantes
datosrules <- datosrules[!is.redundant(datosrules)]
inspect(datosrules)

# Convirtiendo las reglas a un data frame
datosrules_df <- as(datosrules, "data.frame")
str(datosrules_df)

# Guardando las reglas a una archivo CSV 
write(datosrules, file = "datosrules.csv",
      sep = ",", quote = TRUE, row.names = FALSE)


#####################
# EJEMPLO GROCERIES # 
#####################

# En el archivo groceries.csv se tiene almacenados el registro de todas 
# las compras que se han realizado en un supermercado durante 30 dias.

# Paso 1: Cargando los datos

# Cargar la data en un matrix sparse
library(arules)
groceries <- read.transactions("groceries.csv", 
                               rm.duplicates = TRUE,
                               sep = ",")

summary(groceries)

# Tambien es posible mostrar los resultados en formato de dataframe 
df_groceries <- as(groceries, Class = "data.frame")

# Para extraer el tamaño de cada transaccion se emplea la funcion size().
tamanos <- size(groceries)
summary(tamanos)

quantile(tamanos, probs = seq(0,1,0.1))

data.frame(tamanos) %>%
  ggplot(aes(x = tamanos)) +
  geom_histogram(color="black",fill="red") +
  labs(title = "Distribucion del tamaño de las transacciones",
       x = "Tamaño") +
  theme_bw()


# Paso 2: Explorando los datos

inspect(groceries[1])
inspect(groceries[9835])

# Mostrar las 5 primeras transacciones
inspect(groceries[1:5])

# Examinar la frecuencia de items
itemFrequency(groceries[,1:3])

# abrasive cleaner artif. sweetener   baby cosmetics 
#     0.0035587189     0.0032536858     0.0006100661

# abrasive cleaner ha sido encontrado en el 0.3% de las transacciones

# Examinar la frecuencia de todos los items
itemFrequency(groceries)

head(sort(itemFrequency(groceries),decreasing = TRUE),5)

groceries %>% itemFrequency %>% sort(decreasing = TRUE) %>% head(5)

# Graficar la frecuencia de items
itemFrequencyPlot(groceries, 
                  support = 0.1)
                  
itemFrequencyPlot(groceries, 
                  topN = 20,
                  col=brewer.pal(8,'Pastel2'),
                  main='Relative Item Frequency Plot',
                  ylab="Item Frequency (Relative)")

itemFrequencyPlot(groceries, 
                  topN = 20,
                  col=brewer.pal(8,'Pastel2'),
                  main='Relative Item Frequency Plot',
                  ylab="Item Frequency (Absolute)",
                  type = "absolute")


# Paso 3: Encontrar las Reglas de Asociacion
library(arules)

# Reglas de asociacion con los parametros por defecto
apriori(groceries)

# Configurando los niveles de soporte y confianza 
soporte <- 30 / dim(groceries)[1]
reglas <- apriori(data = groceries,
                  parameter = list(support = soporte,
                                   confidence = 0.70,
                                   # Se especifica que se creen reglas
                                   target = "rules"))
reglas

# Paso 6: Evaluando las reglas de asociacion

# Resumiendo las reglas de asociacion
summary(reglas)

# Mostrar la primera regla
inspect(reglas[1])

#     lhs                       rhs          support     confidence lift     count
# [1] {baking powder,yogurt} => {whole milk} 0.003253686 0.7111111  2.783039 32   

# Con un soporte de 0.00305033 y una confianza de 0.7, se puede decir que 
# esta regla cubre el 0.3% de todas las transacciones y se cumple en el 
# 71.1% de todas las compras donde se adquirio banking powdrer y yogurts. 
# El lift nos indica cuan propenso un cliente compre  whole milk con 
# relacion a un cliente promedio, dado que compre baking powder y yogurt. 

# Mostrar las tres primeras reglas
inspect(reglas[1:3])

#----------------------------------------------------------------------
# Paso 5: Visualizando las reglas de asociacion

# Ordenando las reglas por el lift
inspect(sort(reglas, by = "lift")[1:5])

# Ordenando las reglas por el soporte
inspect(sort(reglas, by = "support")[1:5])

# Graficando el soporte y la confianza de las reglas
library(arulesViz)
plot(reglas)

# Representacion Grafica de las Reglas
plot(reglas[1:19],
     method = "graph",
     control = list(type = "items"))

# Encontrando subconjuntos de reglas que contengan algun item "curd"
curdrules <- subset(reglas, items %in% "curd")
inspect(curdrules)


# Paso 6: Filtrado de reglas

# Restringir las reglas que se crean 

soporte <- 30 / dim(groceries)[1]
reglas_vegetables <- apriori(data = groceries,
                             parameter = list(support = soporte,
                                              confidence = 0.70,
                                              target = "rules"),
                             appearance = list(rhs = "other vegetables"))

summary(reglas_vegetables)

inspect(reglas_vegetables)

# Convirtiendo las reglas a un data frame
groceryrules_df <- as(reglas, "data.frame")
str(groceryrules_df)

# Guardando las reglas a una archivo CSV 
write(reglas, file = "groceryrules.csv",
      sep = ",", quote = TRUE, row.names = FALSE)

# FIN !!


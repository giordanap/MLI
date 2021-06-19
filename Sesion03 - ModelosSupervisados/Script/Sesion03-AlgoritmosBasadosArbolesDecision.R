#########################################################################
#########------- Machine Learning Inmersion ------------#################
#########################################################################

# Capacitador: André Omar Chávez Panduro
# email: andrecp38@gmail.com / andre.chavez@urp.edu.pe
# Tema: Arboles Clasificacion: CART - CHAID - C50 - Bagging - Boosting
# version: 2.0
#########################################################################


#---------------------------------------------------------
# Para limpiar el workspace, por si hubiera algun dataset 
# o informacion cargada
rm(list = ls())
dev.off()
options(scipen=999) # Quitar la notacion cientifica

#---------------------------------------------------------
# Cambiar el directorio de trabajo
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

#--------------------------------------------
# Paquetes
library(foreign)
library(gmodels)
library(partykit)
library(rpart)
library(rpart.plot)
library(caTools)
library(caret)
library(ggplot2)
library(MLmetrics)
library(randomForest)
library(ISLR)


#######################
# 1. LECTURA DE DATOS #
#######################

library(foreign)
datos <-read.spss("Churn-arboles.sav",
                  use.value.labels=TRUE, 
                  to.data.frame=TRUE)
str(datos)# Tipos de datos

# No considerar la variable de identificaci?n ID
datos$ID <- NULL
str(datos)

# Etiquetando las opciones de las variables categoricas
levels(datos$CHURN)
levels(datos$SEXO)  <- c("Fem","Masc")
levels(datos$CIVIL) <- c("Casado","Soltero")
# Poner en niveles las categorias de autos, Si - No
levels(datos$AUTO) 
levels(datos$CHURN) <- c("Actual","Fuga")

str(datos)

# Direccionar sus datos al dataframe
attach(datos)

###################################################
# 2. ARBOL DE CLASIFICACION CON EL ALGORITMO CART #
###################################################

library(rpart)

# Verificamos la distribucion de la fuga de clientes
table(CHURN)
prop.table(table(CHURN))

#---------------------------------------------------------------
# Ejemplo 1: Arbol con los parametros por defecto
set.seed(123)

arbol1 <- rpart(CHURN ~ . ,  # Y~x
                data=datos, 
                method="class")
arbol1
# Si usa method="anova" es para Modelos de Regresion

# Graficando el arbol
library(rpart.plot)
windows()
rpart.plot(arbol1, digits=-1, type=0, extra=101,cex = .7, nn=TRUE)
rpart.plot(arbol1, digits=-1, type=1, extra=101,cex = .7, nn=TRUE)
windows()
rpart.plot(arbol1, digits=-1, type=2, extra=101,cex = .7, nn=TRUE)
windows()
rpart.plot(arbol1, digits=-1, type=3, extra=101,cex = .7, nn=TRUE)
windows()
rpart.plot(arbol1, digits=-1, type=4, extra=101,cex = .7, nn=TRUE)

# Mejorando los Graficos
library(partykit)
windows()
plot(as.party(arbol1), tp_args = list(id = FALSE))

#------------------------------------------------------------------
# Ejemplo 2: Arbol controlando parametros
# Parametros 
# minsplit:   Indica el numero minimo de observaciones en un nodo para
#             que este sea dividido. Minimo para que un nodo sea padre. 
#             Esta opcion por defecto es 20.
# minbucket:  Indica el numero minimo de observaciones en cualquier
#             nodo terminal. Por defecto esta opcion es el valor 
#             redondeado de minsplit/3.
# cp:         Parametro de complejidad. Indica que si el criterio de 
#             impureza no es reducido en mas de cp*100% entonces se 
#             para. Por defecto cp=0.01. Es decir, la reduccion en la 
#             impureza del nodo terminal debe ser de al menos 1% de la
#             impureza inicial.
# maxdepth:   Condiciona la profundidad maxima del arbol. 
#             Por defecto esta establecida como 30.

# rpart.control(minsplit = 20, minbucket = round(minsplit/3), cp = 0.01, 
# maxcompete = 4, maxsurrogate = 5, usesurrogate = 2, xval = 10,
# surrogatestyle = 0, maxdepth = 30, ...)

set.seed(123)
arbol2 <- rpart(CHURN ~ . , 
                data=datos,
                control=rpart.control(
                        minsplit=500, 
                        minbucket=200),
                method="class")

windows()
rpart.plot(arbol2, digits=-1, type=2, extra=101, cex = 0.7, nn=TRUE)

#---------------------------------------------------------------------
# Ejemplo 3: Controlando el crecimiento del arbol
# con el parametro de complejidad (cp=0.05)

set.seed(123)
arbol3 <-  rpart(CHURN ~ . , 
                 data=datos,
                 control=rpart.control(cp=0),
                 method="class")
windows()
rpart.plot(arbol3, digits=-1, type=2, extra=101, cex = 0.7, nn=TRUE)

printcp(arbol3)

#----------------------------------------------------------------------
# Ejemplo 4: cp=0.001 para obtener un arbol con mas ramas

set.seed(123)
arbol4 <- rpart(CHURN ~ . ,
                data=datos, 
                method="class",
                cp=0)
windows()
rpart.plot(arbol4, digits=-1, type=2, extra=101, cex = 0.7,  nn=TRUE)

# Debemos elegir el mejor parametro de complejidad
printcp(arbol4)

#--------------------------------------------------------------------
# Ejemplo 5: Recortar el arbol (prune)

arbol5 <- prune(arbol4,cp=0.00292398) # Podas o recortas el arbol

windows()
rpart.plot(arbol5, digits=-1, type=2, extra=101, cex = .7, nn=TRUE)


#----------------------------------------------
# Ejemplo 6: Valor optimo de CP

set.seed(123)
arbol.completo <- rpart(CHURN ~ . ,
                        data=datos,
                        method="class",
                        cp=0, 
                        minbucket=0)
arbol.completo$cptable[,4]

printcp(arbol.completo)

plotcp(arbol.completo)

rpart.plot(arbol.completo, digits=-1, type=2, extra=101, cex = 0.7, nn=TRUE)

# Seleccionando el cp optimo
# arbol.pruned <- prune(arbol.completo,cp=0.00292398)
xerr <- arbol.completo$cptable[,"xerror"]
xerr

minxerr <- which.min(xerr)
minxerr

mincp <- arbol.completo$cptable[minxerr, "CP"]
mincp

arbol.pruned <- prune(arbol.completo,cp=mincp)

printcp(arbol.pruned)
plotcp(arbol.pruned)

rpart.plot(arbol.pruned, type=2, extra=101, cex = 0.7, nn=TRUE)

# Prediccion con Arboles
clase.cart <- predict(arbol.pruned,datos,type="class")    # Prediccion de la clase
proba.cart <- predict(arbol.pruned, datos, type = "prob") # Probabilidad 
head(proba.cart)
proba.cart <- proba.cart[,2] # Selecciono la probabilidad de fuga

datoscart <- cbind(datos,clase.cart, proba.cart)
head(datoscart)

# Usando el arbol para convertir variable numerica a categorica

arbol7 <- rpart(CHURN ~ INGRESO, data=datos, method="class", cp=0.005)

windows()
rpart.plot(arbol7, digits=-1, type=2, extra=101, cex = 0.7, nn=TRUE)         

datos2 <- datos # Hago una copia

datos2$INGRESO.CAT <- cut(datos2$INGRESO, 
                          breaks = c(-Inf,27900,Inf),
                          labels = c("Menos de 27900", "De 27900 a mas"),
                          right = FALSE)

table(datos2$INGRESO.CAT)

prop.table(table(datos2$CHURN))
prop.table(table(datos2$INGRESO.CAT,datos2$CHURN),1)


########################################################
# 3. DIVISION DE LA DATA EN MUESTRA DE TRAINING Y TEST #
########################################################

#-------------------------------------------------------------------
# Seleccion de muestra de entrenamiento (70%) y de prueba (30%)
str(datos)                               # 1345 datos

library(caret)
set.seed(123) 

index      <- createDataPartition(datos$CHURN, p=0.7, list=FALSE)
data.train <- datos[ index, ]            # 943 datos trainig             
data.test  <- datos[-index, ]            # 402 datos testing

# Verificando que la particion mantenga las proporciones de la data
round(prop.table(table(datos$CHURN)),3)
round(prop.table(table(data.train$CHURN)),3)
round(prop.table(table(data.test$CHURN)),3)

# De acuerdo a los indicadores obtenidos y a la teoria estudiada,
# entrenar el mejor algoritmo CART con la data de entrenamiento.

modelo_cart

###########################################################################
# 4. PREDICIENDO LA CLASE Y PROBABILIDAD CON LOS MODELOS EN LA DATA TEST #
###########################################################################

# 1. Prediccion de la clase y probabilidad con CART
CLASE.CART <- predict(modelo_cart,newdata = data.test )
head(CLASE.CART)

PROBA.CART <- predict(modelo_cart,newdata = data.test, type="prob")
PROBA.CART <- PROBA.CART[,2]
head(PROBA.CART)


##############################################################
# 5. EVALUANDO LA PERFOMANCE DE LOS MODELOS EN LA DATA TEST #
##############################################################

#------------------------------------------------------------
# a. Evaluando la performance del modelo CART, indicadores tecnicos.

# Tabla de clasificacion
library(gmodels)
CrossTable(x = data.test$CHURN, y = CLASE.CART,
           prop.t=FALSE, prop.c=FALSE, prop.chisq = FALSE)

addmargins(table(Real=data.test$CHURN,Clase_Predicha=CLASE.CART))
prop.table(table(Real=data.test$CHURN,Clase_Predicha=CLASE.CART),1)

# Calcular el accuracy
accuracy <- mean(data.test$CHURN==CLASE.CART) ; accuracy

# Calcular el error de mala clasificacion
error <- mean(data.test$CHURN!=CLASE.CART) ; error

# Curva ROC usando el paquete caTools
library(caTools)
colAUC(PROBA.CART,data.test$CHURN,plotROC = TRUE)
abline(0, 1,col="red")

# Log-Loss
real <- as.numeric(data.test$CHURN)
real <- ifelse(real==2,1,0)
LogLoss(PROBA.CART,real)

# Matriz de confusion
library(caret)
caret::confusionMatrix(CLASE.CART,data.test$CHURN,positive="Fuga")

# b. Indicadores de Negocio o Comerciales #
#------------------------------------------------------------
library(modelplotr)
scores_and_ntiles <- prepare_scores_and_ntiles(datasets=list("data.train","data.test"),
                                               dataset_labels = list("train data","test data"),
                                               models = list("modelo_cart"),
                                               model_labels = list("Modelo CART"),
                                               target_column="CHURN",
                                               ntiles=10)


plot_input <- plotting_scope(prepared_input = scores_and_ntiles,
                             select_model_label = "Modelo CART",
                             select_dataset_label = "test data")

# Curva o Grafico de Ganancias Acumuladas
# Pregunta de Negocio: 
# ¿Cuántos clientes fugados podemos identificar con el 20% 
# superior de nuestros modelos predictivos?
plot_cumgains(data = plot_input)

# Elegimos entonces el cualtil 20 o decil 2
windows()
plot_cumgains(data = plot_input,highlight_ntile = 4)

# Curva o Grafico de Lift Acumulado
# Pregunta de Negocio: 
# Comentar Caso Empresas sin Analytics
# Cuanto mas o superior es elegir con mi modelo que sin nada.
# plot the cumulative lift plot and annotate the plot at percentile = 20
plot_cumlift(data = plot_input,highlight_ntile = 2)


# Grafico de Respuesta - Decil
# Pregunta de Negocio: 
# Cuanto es el % esperado de fugas en el decil?
plot_response(data = plot_input,highlight_ntile = 3)

# Grafico de Respuesta Acumulada - Decil
# Pregunta de Negocio:  
# Cuanto es el % esperado de las fugas en la seleccion?
plot_cumresponse(data = plot_input,highlight_ntile = 3)

# Podemos graficar todos juntos y discutir
plot_multiplot(data = plot_input,highlight_ntile=2,
               save_fig = TRUE,save_fig_filename = 'Indicadores Negocio Churn Cliente')

# Decisiones financieras con modelos predictivos?
plot_profit(data=plot_input,fixed_costs=75000,variable_costs_per_unit=50,profit_per_unit=200)


##########################
# 6. SELECCION DE VARIABLES #
##########################

# Algoritmo Boruta
library(Boruta)

set.seed(111)

boruta.data <- Boruta(CHURN ~ ., data = datos)

print(boruta.data)

plot(boruta.data,cex.axis=0.5)


####################################################
# 7. ARBOL DE CLASIFICACION CON EL ALGORITMO CHAID #
####################################################

library(partykit)
set.seed(100)
# Entrenamos el arbol chaid sobre la data de train
arbol_chaid <- ctree(CHURN~ .,data = data.train,
                     control=ctree_control(mincriterion=0.50))

# Validamos el arbol chaid
clase_arbol=predict(arbol_chaid, newdata= data.test,type="response")
clase_arbol

# Validacion de indicadores tecnicos
tablaChaid=confusionMatrix(clase_arbol, data.test$CHURN ,positive = "Fuga")
tablaChaid

# Grafico del arbol
windows()
plot(arbol_chaid,type='extended') 

####################################################
# 8. ARBOL DE CLASIFICACION CON EL ALGORITMO C50 #
####################################################
library(C50)
modeloc50 <- C5.0(CHURN~.,data = data.train,
                  trials = 50, # Numero de arboles
                  rules= TRUE, # Reglas de seleccion
                  tree=TRUE,  
                  winnow=TRUE) # Seleccion de variables

probaC50=predict(modeloc50, newdata=,data.test,
                 type="prob")[,2] # Dame la segunda columna, es decir 
# la probabilidad del default
predC50=predict(modeloc50, newdata=,data.test,
                type="class")

# Ejercicio hallar la matriz de clasificacion.
matrizC <- confusionMatrix(predC50,data.test$CHURN,positive = "Fuga")
matrizC


library(pROC)
auc <- roc()
gini <- 2*(auc$auc)-1
gini 

#----------------------------------------------------------------------------
############################################################
# 9. ARBOL DE CLASIFICACION CON EL ALGORITMO RANDOM FOREST #
############################################################

set.seed(1234)
library(randomForest)
modelo_rf <- randomForest(CHURN~., # Y ~ X
                          data = ,   # Datos a entrenar 
                          ntree=200,           # Numero de arboles
                          mtry = 3,            # Cantidad de variables
                          # Raiz2 Total de variables o 40%-60% total variables.
                          importance = TRUE,   # Determina la importancia de las variables
                          replace=T) 
#----------------------------------
# OOB error (out of bag error)
#
print(modelo_rf)

# Graficar Error del Modelo
#
# En este grafico se muestra un modelo que intenta predecir 
# la variable churn={FUGA,ACTUAL}. 
# La linea negra representa el OOB, 
# la linea roja es el error al intentar predecir churn={ACTUAL}, 
# la linea verde es el error en la prediccion churn={FUGA}. 
# La linea negra siempre sera el OOB, y las siguientes lineas
# se pueden identificar con la matriz de confusion 
# usando print(MODELO.RF) 

plot(modelo_rf)

#-----------------------------------------------------------------------------
# Importancia de las variables
# La tabla MeanDecreaseAccuracy representa en cuanto removiendo 
# dicha variable se reduce la precision del modelo.
# Un valor mas alto de MeanDecreaseAccuracy o 
# del MeanDecreaseGiniScore, implica una mayor importancia 
# de la variable en el modelo.

varImpPlot(modelo_rf)
modelo_rf$importance


# Predecimos sobre la datatest
proba_rf <- predict(modelo_rf, # Modelo entrenado de RF
                    newdata=,
                    type="prob")

# Debemos tener cuidado con las probabilidades , pues tenemos hasta 2.
# Curva ROC
library(pROC)
AUC <- roc(, proba_rf) 
auc_rf=AUC$auc
auc_rf
# Indice de gini
gini_rf <- 2*(AUC$auc) -1
gini_rf

# Calcular los valores predichos
PRED_rf <-predict(modelo_rf,,type="class")

# Calcular la matriz de confusion
tabla_rf=confusionMatrix(PRED_rf,,
                         positive = "Fuga")
tabla_rf

# Boosting Adabag
library(adabag)
set.seed(1234)
# 'Freund', 'Breiman', 'Zhu' # 
modelo_boosting<-boosting,        # Y~ x
data = ,
coeflearn='Breiman', # Learning rate
boos=TRUE, 
mfinal=100) # Numero de arboles

Importancia_Varia <- data.frame(modelo_boosting$importance)
write.csv(Importancia_Varia,"Importancia_Variables.csv")

# Predecimos
proba_boosting<-predict(modelo_boosting,)
prob_boosting <- proba_boosting$prob[,1]

# Curva ROC
AUC <- roc(, prob_boosting) 
auc_boosting=AUC$auc

# Indice de gini


# Calcular los valores predichos
PRED_boosting <-proba_boosting$class
PRED_boosting <- as.factor(PRED_boosting) # Convierto a factor

# Calcular la matriz de confusion
tabla=confusionMatrix(PRED_boosting,imbal_testing$Fuga,
                      positive = "Si_Fuga")
tabla


##############################################################
# 6. GUARDAMOS EL MODELO ENTRENADO PARA USARLO CON DATA NUEVA #
##############################################################

# Guardar el modelo
saveRDS(modelo_cart,"ArbolCart.rds")

#------------------------------------------------------
# PARTE 2: DESPLIEGUE Y PRODUCTIVO DE MODELOS
#------------------------------------------------------

# Deseamos replicar o implementar el modelo.
# Leemos el modelo predictivo.
RegresionLog <- readRDS("ArbolCart.rds")

# Leemos el dataset de nuevos leads
library(foreign)
datos_n <-read.spss("Churn-nuevos-arboles.sav",
                    use.value.labels=TRUE, 
                    to.data.frame=TRUE)

# Decodificacion

levels(datos_n$SEXO)  <- c("Fem","Masc")
levels(datos_n$CIVIL) <- c("Casado","Soltero")
levels(datos_n$AUTO)  <- c("Si","No")

# Ejercicio calificado:

# Dado que tenemos algoritmos entrenados y validados, el area
# comercial de la empresa nos ha solicitado scorear o puntuar 
# la leads que van a gestionarse en las campanas comerciales.

# Como se han enterado que el algoritmo de CART es uno
# de los mas sofisticados ha pedido explicitamente su uso.

# Entregables:
# Base completa de leads nuevos, probabilidad de fuga y la clase 
# pronosticada.


# FIN !!

#########################################################################
#########------- Machine Learning Inmersion ------------#################
#########################################################################

# Capacitador: André Omar Chávez Panduro
# email: andrecp38@gmail.com / andre.chavez@urp.edu.pe
# Tema: Arboles Clasificacion: Bagging - Boosting / Balanceo de Datos
# version: 2.0
#########################################################################

#---------------------------------------------------------
# Para limpiar el workspace, por si hubiera algun dataset 
# o informacion cargada
rm(list = ls())
dev.off()

#---------------------------------------------------------
# Cambiar el directorio de trabajo
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

#----------------------------------------------------------
# Paquetes
library(caret)    # Modelamiento Datos    
library(DMwR)     # Balance de Muestras
library(caTools)  # Indicadores de validacion
library(ggplot2)  # Graficos

##############################################################
#  Prediccion de fuga de clientes de una entidad financiera  #
##############################################################

# Se tienen de un año del producto CTS (compensacion por tiempo de 
# servicios) en todas las agencias de una entidad financiera. 
#
# Variables predictoras:

# Tasa      	      Tasa de interes de la cuenta CTS
# Saldo_soles	      Monto de Saldo de la cuenta CTS, en Soles.
# Edad	              Edad del cliente en anos
# EstadoCivil	      Estado Civil: Div.Sol.Viu = Divorciado, Soltero y Viudo y 
#                                 Cas.Conv = Casado, Conviviente
# Region	          Zona a la que pertenece el cliente: 
#                   NORTE.SUR, ORIENTE, CENTRO, LIMA_CALLAO
# CrossSell           Numero de productos vigentes con el banco, tanto pasivos o 
#                   activos
# Ratio.Ant         Ratio Ant_Cts / Ant_Banco  
#     Ant_Banco	        Tiempo de antiguedad del cliente (en meses)
#     Ant_Cts	          Tiempo de antiguedad de la cuenta CTS (en meses)
#
# Variable dependiente:
# Fuga              0 = cliente no fugado, 1 = cliente fugado

#---------------------------------------------------------------------------
# Lectura de datos 
datos.m <- read.csv("Fuga_Ahorros.csv",sep=";",dec=",")
str(datos.m)

# No considerar el campo id
datos.m$Id <- NULL

# Convertir a factor y recodificar la variable Fuga
datos.m$Fuga <- as.factor(datos.m$Fuga)
levels(datos.m$Fuga)  <- c("No_Fuga","Si_Fuga")
str(datos.m)

# Observamos la distribucion del target
table(datos.m$Fuga)
round(prop.table(table(datos.m$Fuga))*100,2)

#-------------------------------------------------------------------
# Seleccion de muestra de entrenamiento (80%) y de prueba (20%)
library(caret)
set.seed(123) 
index   <- createDataPartition(datos.m$Fuga, p=0.7, list=FALSE)
imbal_train    <- datos.m[ index, ] # Data de entrenamiento
imbal_testing  <- datos.m[-index, ] # Data de test

# Verificando que se mantenga la proporcion original
addmargins(table(datos.m$Fuga))
round(prop.table(table(datos.m$Fuga))*100,2)

addmargins(table(imbal_train$Fuga))
round(prop.table(table(imbal_train$Fuga))*100,2)

addmargins(table(imbal_testing$Fuga))
round(prop.table(table(imbal_testing$Fuga))*100,2)

#---------------------------------------------------------------------
# Generar diferentes versiones de balanceo de la muestra de training**
# Siempre el balance se hace sobre el entrenamiento, el testing no se toca.
# Antes del modelamiento
  
# Undersampling
# El algoritmo de balanceo necesita entender cual es la clase minoritaria
# y cual es la mayoritaria
set.seed(123) # Semilla aleatoria
under_train <- downSample(x = imbal_train[,c(1:7)],  # X
                          y = imbal_train$Fuga,      # y
                          yname="Fuga")

addmargins(table(under_train$Fuga)) # Valores absolutos
prop.table(table(under_train$Fuga)) # Valores relativos

# OverSampling
set.seed(123)
over_train <- upSample(x = imbal_train[, c(1:7)], # X
                     y = imbal_train$Fuga,        # y
                     yname="Fuga")
                         
addmargins(table(over_train$Fuga))
prop.table(table(over_train$Fuga)) # Valores relativos

# Smote 
library(DMwR)
set.seed(123)
smote_train <- SMOTE(Fuga ~ .,        # Formula
                     data=imbal_train,# Dataset train
                     perc.over = 200, # Aumento
                     perc.under=150)  # Reduccion              

addmargins(table(imbal_train$Fuga))  # Train real
addmargins(table(smote_train$Fuga) ) # Train Smote

# No_Fuga Si_Fuga 
#   4201     226 
# perc.over  = 200   significa que va a adicionar 2 veces (200%) a la 
#                    clase minoritaria 226 + 2*226 = 678
# perc.under = 150   significa que por cada caso adicionado (2*226=452) 
#                    escoger? el 150% (678) de la clase mayoritaria
# No_Fuga Si_Fuga 
#     678     678 
                     
#---------------------------------------------------------
# Entrenar el modelo con validacion cruzada 
# Usar como indicador el Accuracy

# Relacion de modelos 
library(caret)
names(getModelInfo())

# Relacion de parametros a ajustar de un modelo
modelLookup(model='glm')

# Metodo de Validacion para todos los modelos  
ctrl <- trainControl(method="cv", number=10)

# Para usar el AROC en vez del Accuracy durante la validaci?n
# classProbs=TRUE,summaryFunction = twoClassSummary
# en el train retirar la opci?n metric = )

# Para usar el LogLoss en ver del Accuracy durante la validaci?n
# classProbs=TRUE,summaryFunction = mnLogLoss)
# en el train usar  metric="logLoss" )

# 1. Modelo con lo datos originales (desbalanceados)
set.seed(123)
modelo_orig   <- caret::train(Fuga ~ ., 
                    data = imbal_train, 
                    method="glm", 
                    family="binomial", 
                    trControl = ctrl,
                    metric="Accuracy")

modelo_orig

summary(modelo_orig)

# 2. Modelo con los datos balanceados (undersampling)
set.seed(123)
modelo_under  <- caret::train(Fuga ~ ., 
                       data = under_train, 
                       method="glm", 
                       family="binomial", 
                       trControl = ctrl, 
                       metric="Accuracy")
modelo_under

# 3. Modelo con los datos balanceados (oversampling)
set.seed(123)
modelo_over    <- caret::train(Fuga ~ ., 
                        data = over_train, 
                        method="glm",
                        family="binomial", 
                        trControl = ctrl, 
                        metric="Accuracy")
modelo_over

# 4. Modelo con los datos balanceados (SMOTE)
set.seed(123)
modelo_smote      <- caret::train(Fuga ~ ., 
                      data = smote_train, 
                      method="glm", 
                      family="binomial", 
                      trControl = ctrl, 
                      metric="Accuracy")
modelo_smote

#-------------------------------------------------------------
# Comparando la muestras para los cuatro modelos
modelos  <- list(original = modelo_orig,  # Data Desbalanceada
                 under    = modelo_under,
                 over     = modelo_over,
                 SMOTE    = modelo_smote)

comparacion_modelos <- resamples(modelos)
summary(comparacion_modelos)

dotplot(comparacion_modelos)
bwplot(comparacion_modelos)

#----------------------------------------------------------------------------
# Prediccion de los modelos en la data testing
prop.table(table(imbal_testing$Fuga)) # Distribucion desbalanceada

# 1. Prediccion del modelo_orig en la data testing
clase.modelo_orig <- predict(modelo_orig,newdata = imbal_testing )
proba.modelo_orig <- predict(modelo_orig,newdata = imbal_testing, type="prob")
proba.modelo_orig <- proba.modelo_orig[,2]

library(caTools)
colAUC(proba.modelo_orig,imbal_testing$Fuga,plotROC = TRUE)
abline(0, 1,col="red")

result1 <- caret::confusionMatrix(clase.modelo_orig,
                                 imbal_testing$Fuga,
                                 positive="Si_Fuga")
result1$byClass["Sensitivity"] 
result1$byClass["Specificity"] 
result1$overall["Accuracy"]

result1

# 2. Prediccion del modelo_under en la data testing
clase.modelo_under <- predict(modelo_under,newdata = imbal_testing )
proba.modelo_under <- predict(modelo_under,newdata = imbal_testing, type="prob")
proba.modelo_under <- proba.modelo_under[,2]

library(caTools)
colAUC(proba.modelo_under,imbal_testing$Fuga,plotROC = TRUE)
abline(0, 1,col="red")

result2 <- caret::confusionMatrix(clase.modelo_under,
                                 imbal_testing$Fuga,
                                 positive="Si_Fuga")
result2$byClass["Sensitivity"] 
result2$byClass["Specificity"] 
result2$overall["Accuracy"]
result2

# 3. Prediccion del modelo_over en la data testing
clase.modelo_over  <- predict(modelo_over,newdata = imbal_testing )
proba.modelo_over  <- predict(modelo_over,newdata = imbal_testing, type="prob")
proba.modelo_over  <- proba.modelo_over[,2]

library(caTools)
colAUC(proba.modelo_over,imbal_testing$Fuga,plotROC = TRUE)
abline(0, 1,col="red")

result3 <- caret::confusionMatrix(clase.modelo_over,
                                 imbal_testing$Fuga,
                                 positive="Si_Fuga")
result3$byClass["Sensitivity"] 
result3$byClass["Specificity"] 
result3$overall["Accuracy"]
result3

# 4. Prediccion del modelo_smote en la data testing
clase.modelo_smote <- predict(modelo_smote,newdata = imbal_testing )
proba.modelo_smote <- predict(modelo_smote,newdata = imbal_testing, type="prob")
proba.modelo_smote <- proba.modelo_smote[,2]

library(caTools)
colAUC(proba.modelo_smote,imbal_testing$Fuga,plotROC = TRUE)
abline(0, 1,col="red")

result4 <- caret::confusionMatrix(clase.modelo_smote,
                                 imbal_testing$Fuga,
                                 positive="Si_Fuga")
result4$byClass["Sensitivity"] 
result4$byClass["Specificity"] 
result4$overall["Accuracy"]
result4

#----------------------------------------------------------------------------
#######################################################
###########  BAGGING ARBOLES : RANDOM FOREST ##########
#######################################################

set.seed(1234)
library(randomForest)
modelo_rf <- randomForest(Fuga~., # Y ~ X
                          data = under_train,   # Datos a entrenar 
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
# En este gr?fico se muestra un modelo que intenta predecir 
# la variable churn={FUGA,ACTUAL}. 
# La linea negra representa el OOB, 
# la linea roja es el error al intentar predecir churn={ACTUAL}, 
# la linea verde es el error en la prediccion churn={FUGA}. 
# La linea negra siempre ser? el OOB, y las siguientes lineas
# se pueden identificar con la matriz de confusi?n 
# usando print(MODELO.RF) 

plot(modelo_rf)

#-----------------------------------------------------------------------------
# Importancia de las variables
# La tabla MeanDecreaseAccuracy representa en cu?nto removiendo 
# dicha variable se reduce la precisi?n del modelo.
# Un valor m?s alto de MeanDecreaseAccuracy o 
# del MeanDecreaseGiniScore, implica una mayor importancia 
# de la variable en el modelo.

varImpPlot(modelo_rf)
modelo_rf$importance


# Predecimos sobre la datatest
proba_rf <- predict(modelo_rf, # Modelo entrenado de RF
                  newdata=imbal_testing,
                  type="prob")

proba_rf <- proba_rf[,2]
# Curva ROC
library(pROC)
AUC <- roc(imbal_testing$Fuga, proba_rf) 
auc_rf=AUC$auc
auc_rf
# Indice de gini
gini_rf <- 2*(AUC$auc) -1
gini_rf

# Calcular los valores predichos
PRED_rf <-predict(modelo_rf,imbal_testing,type="class")

# Calcular la matriz de confusion
tabla=confusionMatrix(PRED_rf,imbal_testing$Fuga,
                      positive = "Si_Fuga")
tabla

# Boosting Adabag
library(adabag)
set.seed(1234)
# 'Freund', 'Breiman', 'Zhu' # 
modelo_boosting<-boosting(Fuga~.,
                          data = under_train,
                          coeflearn='Breiman', # Learning rate
                          boos=TRUE, 
                          mfinal=100) # Numero de arboles

Importancia_Varia <- data.frame(modelo_boosting$importance)
write.csv(Importancia_Varia,"Importancia_Variables.csv")

# Predecimos
proba_boosting<-predict(modelo_boosting,imbal_testing)
prob_boosting <- proba_boosting$prob[,1]

# Curva ROC
AUC <- roc(imbal_testing$Fuga, prob_boosting) 
auc_boosting=AUC$auc

# Indice de gini
gini_boosting <- 2*(AUC$auc) -1

# Calcular los valores predichos
PRED_boosting <-proba_boosting$class
PRED_boosting <- as.factor(PRED_boosting) # Convierto a factor

# Calcular la matriz de confusion
tabla=confusionMatrix(PRED_boosting,imbal_testing$Fuga,
                      positive = "Si_Fuga")
tabla
#############################################
#  BAGGING CON CARET Y VALIDACION CRUZADA ###
#############################################

# Relacion de parametros a ajustar de un modelo
modelLookup(model='treebag')

# Aplicando el modelo con Validacion Cruzada 
ctrl <- trainControl(method="cv", number=10)

set.seed(123)
modelo_bag <- train(Fuga ~ ., 
                    data = imbal_train, 
                    method = "treebag",
                    trControl = ctrl, 
                    tuneLength = 5, 
                    metric="Accuracy")
modelo_bag

plot(modelo_bag)

varImp(modelo_bag)

# treebag no tiene par?metros para hacer tunning

###################################################
# 9. RANDOM FOREST CON CARET Y VALIDACION CRUZADA #
###################################################

# Relacion de parametros a ajustar de un modelo
modelLookup(model='rf')

# Aplicando el modelo con Validacion Cruzada
ctrl <- trainControl(method="cv", number=10)

set.seed(123)
modelo_rf <- train(Fuga ~ ., 
                   data = under_train, 
                   method = "rf", 
                   trControl = ctrl, 
                   tuneLength = 5,
                   metric="Accuracy")
modelo_rf

plot(modelo_rf)

varImp(modelo_rf)

# FIN !!
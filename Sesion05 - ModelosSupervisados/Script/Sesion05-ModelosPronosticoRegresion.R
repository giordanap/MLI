rm(list=ls())
################################################################################
###########---------- Machine Learning Inmersion ------------------#############
################################################################################

# Capacitador: André Omar Chávez Panduro
# email: andrecp38@gmail.com
# Tema: Modelos de Pronostico : Analisis de Regresion
# version: 1.0
#########################################################################
###################################################################

#---------------------------------------------------------
# Para limpiar el workspace, por si hubiera algun dataset 
# o informacion cargada
rm(list = ls())
dev.off()
options(scipen=999) # Desactivar la notacion cientifica

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

# Lectura de datos
datos<- read.csv("bostonvivienda.csv",
                  na.strings = c(""," ",NA)) # Leer la data de entrenamiento

#--------------------------------------------------------
# 1. Deteccion de valores perdidos

# Deteccion de valores perdidos con el paquete DataExplorer
library(DataExplorer)
plot_missing(datos) 

#--------------------------------------------------------
# 2. Analisis Bivariado de la data

correlacion<-cor(datos)

library(corrplot)
corrplot(correlacion, method="number", type="upper")
library("PerformanceAnalytics")
chart.Correlation(datos, histogram=TRUE, pch=19)
library(psych)
pairs.panels(datos, scale=TRUE)
library(corrplot)
corrplot.mixed(cor(datos), order="hclust", tl.col="black")
library(GGally)
ggpairs(datos)
ggcorr(data, nbreaks=8, palette='RdGy', label=TRUE, label_size=5, label_color='white')
library(ggcorrplot)
ggcorrplot(cor(datos), p.mat = cor_pmat(mtcars), hc.order=TRUE, type='lower')


#--------------------------------------------------------
# 3. Colinealidad o Multicolinealidad
altaCorr <- findCorrelation(correlacion, cutoff = .60, names=TRUE)
altaCorr


#-------------------------------------------------------------------
# 4. Seleccion de muestra de entrenamiento (70%) y de prueba (30%)
str(datos)                              

library(caret)
set.seed(123) 

index      <- createDataPartition(datos$crim, p=0.7, list=FALSE)
data.train <- datos[ index, ]            # 943 datos trainig             
data.test  <- datos[-index, ]            # 402 datos testing

#-------------------------------------------------------------------
# 5. Modelos Parametricos 

# Y ~ X Regresion Lineal Simple
reg <- lm(medv ~ lstat, data = data.train)

# Revisamos el resumen del algoritmo
summary(reg)

# El objetivo final es el pronostico o prediccion
pred <- predict(reg,data.test)

# Comparamos los valores reales y predichos
library(forecast)
accuracy(data.test$crim,pred)


# Y ~ X Regresion Lineal Multiple
reg_multiple <- lm(crim ~ ., data = data.train)

# Revisamos el resumen del algoritmo para hacer la seleccion de las mejores variables ...
summary(reg_multiple)
# Ejecutamos el algoritmo con las variables significativas
reg_multiple <- lm(crim ~  rad + medv, data = data.train)

# Vuelvo a revisar el resumen de las variables
summary(reg_multiple)

# Predecir con el modelo parsimonioso

pred_reg_mult <- predict(reg_multiple,data.test)

# Comparamos los valores reales y predichos
library(forecast)
accuracy(data.test$crim,pred)          # Modelo Y ~ X
accuracy(data.test$crim,pred_reg_mult) # Modelo Y ~ X1 + X2

# Metodologia DMC

Comparacion <- data.frame(data.test$crim,pred_reg_mult)
write.csv(Comparacion,"Comparativa_Mod_Regresion.csv")
#-------------------------------------------------------------------
# Regresiones Penalizadas 

# Convierto a una matriz y separo mis X's de Y en el entrenamiento

y_train=as.matrix(data.train$crim)       # Variable y
X_train =as.matrix(data.train[,2:13])     # Covariables

# Convierto a una matriz y separo mis X's de Y en el test
y_test=as.matrix(data.test$crim)
X_test=as.matrix(data.test[,2:13])


# Regresion Penalizadas Ridge
library(glmnet)

fitridge=glmnet(X_train,y_train, # X,y,
                alpha=0) # alpha= 0 Ridge,alpha=1 Lasso,alpha=0.5 Elstic Net

fitridge$beta
plot(fitridge) # las q se alejan mas son las mas importantes

# Encontramos los mejores coeficientes
# Ajustamos un modelo con cv
foundrigde=cv.glmnet(X_train,y_train,
                     alpha=0,
                     nfolds=10)

plot(foundrigde) # con landa de log de 4 a 6 se estabiliza
attributes(foundrigde)
foundrigde$lambda

# Elijo el parametro de regularización
foundrigde$lambda.1se # Muestra el parámetro optimo (agresivo)
foundrigde$lambda.min # Muestra parámetro comercial

# Revisar la contraccion de coeficientes
coef(fitridge,s=foundrigde$lambda.1se) # Ideal x CV
coef(fitridge,s=foundrigde$lambda.min)

# Predecir con la regresion ridge, con el lambda agresivo
prediridge=predict(foundrigde,X_test,s="lambda.1se")

# Comparamos los valores reales y predichos
library(forecast)
accuracy(data.test$crim,prediridge)


# Regresion Penalizadas Lasso

fitlasso=glmnet(X_train,y_train,
                alpha=1)

## Encontrar los mejores coeff
foundlasso=cv.glmnet(X_train,
                    y_train,alpha=1,nfolds=10) 

plot(foundlasso)
foundlasso$lambda.1se # muestra el landa optimo sugerencia
foundlasso$lambda.min 

coef(fitlasso,s=foundlasso$lambda.1se) # Agresivo
coef(fitlasso,s=foundlasso$lambda.min) # Permisivo

# Predecir con la regresion lasso, con el lambda permisivo
predilasso=predict(foundlasso,X_test,s="lambda.1se")

# Comparamos los valores reales y predichos
library(forecast)
accuracy(data.test$crim,predilasso)


## modelo Regresion mediante Elastic Net

fitnet=glmnet(X_train,
              y_train,alpha=0.5)## aplha 0.5 es elasticnet

## Encontrar los mejores coeff

founnet=cv.glmnet(X_train,y_train,
                  alpha=0.5,
                  nfolds=10) 
plot(founnet)
founnet$lambda.1se # muestra el landa optimo sugerencia
founnet$lambda.min # permisivo

coef(fitnet,s=founnet$lambda.1se)
coef(fitnet,s=founnet$lambda.min)

# Prediccion
predinet=predict(founnet,X_test,s="lambda.1se")

# Comparamos los valores reales y predichos
library(forecast)
accuracy(data.test$crim,predinet)


#-------------------------------------------------------------------
# 5. Modelos No Parametricos 
#----------------------------------------------------------------------

# Arboles de Decision : CART

set.seed(123)
library(rpart)
arbol_reg <- rpart(crim ~ . ,
                data=data.train, 
                method="anova")

# Prediccion con Arboles
reg.cart <- predict(arbol_reg,data.test)    # Prediccion de la clase

# Comparamos los valores reales y predichos
library(forecast)
accuracy(data.test$crim,reg.cart)


# Ensamble de Arboles : Random Forest
library(randomForest)
library(ISLR)

set.seed(123)
randomforest_reg <- randomForest( crim ~ ., data = data.train,   # Datos a entrenar 
                           ntree=100,                 # Numero de arboles
                           mtry = 3,                  # Cantidad de variables
                           importance = TRUE,         # Determina la importancia de las variables
                           replace=T)  

# Prediccion con Ensamble de Arboles
reg.rf <- predict(randomforest_reg,data.test)    # Prediccion de la clase

# Comparamos los valores reales y predichos
library(forecast)
accuracy(data.test$crim,reg.rf)

# Boosting de Arboles : XgBoost

library(xgboost)

# Separar covariables y target
Mtrain_XGB <- model.matrix(~ ., data=data.train[,c(2:13)]) # X
Ytrain <- as.vector(data.train$crim)                       # Y

# Construimos una matriz Xgboost
dtrain <- xgb.DMatrix(Mtrain_XGB, label = Ytrain)

# Lo mismo que le hago al train, le hago al test
Mtest_XGB <- model.matrix(~ ., data=data.test[,c(2:13)])
Ytest <- as.vector(data.test$crim)
dtest <- xgb.DMatrix(Mtest_XGB, label = Ytest)

#Hacemos nuestra lista de particiones de datos
watchlist <- list(train = dtrain, test = dtest)

# Tuneamos los parametros de una manera apropiada
param <- list(booster = "gbtree", 
              objective = "reg:linear", 
              eta=0.3,
              alpha=0.8,
              gamma=0.7, 
              max_depth=1, 
              #min_child_weight=1, 
              subsample=0.6, 
              colsample_bytree=0.6,
              eval_metric = "rmse")

xgb_fit <- xgb.train(param, 
                     dtrain, 
                     nround = 1000, 
                     watchlist,verbose = 1,
                     early_stopping_rounds = 15)

# Importancia de las variables
importance_xgb <- xgb.importance(feature_names = colnames(Mtrain_XGB),
                                 model = xgb_fit)
importance_xgb

# Plot de las variables
xgb.plot.importance(importance_matrix = importance_xgb)

# Prediccion
xgb_reg <- predict(xgb_fit,dtest)

# Comparamos los valores reales y predichos
library(forecast)
accuracy(data.test$crim,xgb_reg)

# FIN!
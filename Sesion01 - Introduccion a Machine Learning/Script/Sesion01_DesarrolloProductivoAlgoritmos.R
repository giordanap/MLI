#########################################################################
#########------- Machine Learning Inmersion ------------#################
#########################################################################

# Capacitador: André Omar Chávez Panduro
# email: andrecp38@gmail.com
# Modelos Introduccion a Machine Learning
# version: 1.0
#########################################################################

#---------------------------------------------------------
# Para limpiar el workspace, por si hubiera algun dataset 
# o informacion cargada
rm(list = ls())

#---------------------------------------------------------
# Cambiar el directorio de trabajo
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()

#---------------------------------------------------------
#  Paquetes

library(gplots)
library(vcd)
library(foreign)
library(gmodels)
library(ggplot2)

#############################################
#  SUSCRIPCION: Adquisicion de Clientes  ####
#############################################

# El objetivo es predecir clientes propensos a adquirir o suscribirse 
# algun nuevo producto lanzado en una campana comercial en una empresa de 
# telecomunicaciones.
#
# Se cuenta con una data de 4521 clientes de una empresa de 
# telecomunicaciones donde algunos se suscribieron a una campana comercial 
# y otros no se han suscrito.
#
# Variable Dependiente:  
#           y   (0=No Suscrito, 1=Suscrito)
#
# Variables Independentes:
#           edad        (Edad del cliente en anos)  
#           trabajo     (Tipo de trabajo del cliente)
#           estciv      (Estado civil del cliente) 
#           educacion   (Nivel educativo del cliente)
#           mora        (Tiene algun producto en mora?)
#           balance     (Valor o puntuacion del cliente)  
#

#----------------------------------------------
# PARTE 1: DESCRIPTIVA
#----------------------------------------------

#----------------------------------------------------------
# Preparacion de los datos

library(data.table)

datos <-fread("bancomark.txt",
                header=T, 
                verbose =FALSE, 
                stringsAsFactors=TRUE,
                showProgress =TRUE)

str(datos)

# Etiquetando las opciones de las variables categoricas
levels(datos$y)
levels(datos$y) <- c("No Suscrito","Suscrito")

str(datos)

attach(datos) # Direcciona los analisis al set de datos

#--------------------------------------------
# ENTENDIMIENTO Y VISUALIZACION DE LOS DATOS
#--------------------------------------------

#----------------------------------------
# Analisis descriptivo univariado de variables continuas
summary(y) # Resumen de las variables

round(prop.table(table(y)),2)

summary(dia)

windows()
boxplot(edad,       ylab="Edad",                  col="blue")
boxplot(duracion,   ylab="Duracion del contacto", col="#9999CC")

# Ejercicios
# Construya un boxplot para la variable balance usando boxplot()



#------------------------------------------------------------------
# Analisis descriptivo bivariado de las variables categoricas
prop.table(table(estciv, y),margin=1)
prop.table(table(educacion, y),margin=1)

# Ejercicios
# Construya una tabla de contigencia de CONTACTO,MORA vs. SUSCRIPCION

prop.table(table(contacto, y),margin=1)



library(gmodels)
CrossTable(estciv,
           y,
           prop.t=FALSE,
           prop.r=TRUE,
           prop.c=FALSE,
           prop.chisq=FALSE)

CrossTable(educacion,
           y,
           prop.t=FALSE,
           prop.r=TRUE,
           prop.c=FALSE,
           prop.chisq=FALSE)

# Ejercicios
# Construya un tabla de contingencia de CONTACTO vs. SUSCRIPCION usando CrossTable()






#------------------------------------------------------------------------
#  Visualizacion de una Tabla de Contingencia usando la funcion mosaic()  
library(vcd)

mosaic(y ~ estciv, 
       main = "Estado Civil vs Suscripcion", 
       data=datos, shade=TRUE)

mosaic(y ~ educacion, 
       main = "Nivel educativo vs Suscripcion", 
       data=datos, shade=TRUE)

# Ejercicios
# Presente un grafico de mosaico de CONTACTO, VIVIENDA y PRESTAMO vs SUSCRIPCION





#------------------------------------------------------------------------
#  Visualizacion de una Tabla de Contingencia usando una Matriz Grafica  
#  Grafico de Balloonplots (Grafico de Globos)

Tabla1 <- round(prop.table(table(estciv,y),margin=1),3)
Tabla1

Tabla2 <- round(prop.table(table(educacion,y),margin=1),3)
Tabla2

Tabla3 <- round(prop.table(table(contacto,y),margin=1),3)
Tabla3

library(gplots)

balloonplot(t(Tabla1), 
            main ="Tabla de Contingencia",
            xlab ="Cliente", 
            ylab="Estado Civil",
            label = FALSE, 
            cum.margins=FALSE, 
            label.lines=FALSE,
            show.margins = FALSE)

balloonplot(t(Tabla3), 
            main ="Tabla de Contingencia",
            xlab ="Cliente", 
            ylab="Contacto",
            label = FALSE, 
            cum.margins=FALSE, 
            label.lines=FALSE,
            show.margins = FALSE)

# Ejercicios
# Presente un grafico de balloonplot de CONTACTO, VIVIENDA y PRESTAMO vs SUSCRIPCION




#------------------------------------------------------------------------
#  Visualizacion de una Tabla de Contingencia usando   
#  Grafico de Barras Apiladas (stacked)

library(ggplot2)

Tabla1 <- round(prop.table(table(contacto,y),margin=1),3) ; Tabla1
Tabla1 <-as.data.frame(Tabla1)
str(Tabla1)

ggplot(data=Tabla1, aes(x=contacto, y=Freq, fill=y ) ) +
  geom_bar(stat='identity') +
  theme_bw() +
  labs(title ="Situacion de la Suscripcion segun el contacto", 
       x="Contacto", 
       y= "Frecuencia") + 
  geom_text(aes(label=Freq), 
            position = position_stack(vjust = 0.5),color="black") +
  scale_fill_manual(values=c("darkolivegreen3", "firebrick2")) 


Tabla2 <- round(prop.table(table(vivienda,y),margin=1),3);Tabla2
Tabla2 <-as.data.frame(Tabla2)
str(Tabla2)

ggplot(data=Tabla2, aes(x=vivienda, y=Freq, fill=y ) ) +
  geom_bar(stat='identity') +
  theme_bw() +
  labs(title ="Situacion de la Suscripcion segun el tipo de vivienda", 
       x="Tipo de vivienda", 
       y= "Frecuencia") + 
  geom_text(aes(label=Freq), 
            position = position_stack(vjust = 0.5),color="black") +
  scale_fill_manual(values=c("darkolivegreen3", "firebrick2")) 

# Ejercicios
# Presente un grafico apilado de SUSCRIPCION vs. PRESTAMO o ESTADOCIVIL usando ggplot(). 
# Cree previamente una distribucion bivariada de datos.









#--------------------------------------------------------------
# Analisis descriptivo bivariado de las variables cuantitativas

round(tapply(edad,y,mean),3) 
round(tapply(balance,y,mean),3) 


# Ejercicios
# Presente la duracion de llamadas segun la SUSCRIPCION con redondeo a 2 decimales



#--------------------------------------------------------------
# Analisis grafico bivariado de las variables cuantitativas
# Grafico de barras de variables cuantitativas segun SUSCRIPCION

#-----------------------------------------------------------
# Usando el paquete graphics

# Para el Numero de contactos utilizados (campana)
promedio2 <-  round(tapply(campana,y,mean),2)
promedio2
bar2 <- barplot(promedio2,
                main="Grafico de Barras: Numero promedio de contactos segun suscripcion",
                xlab="Situacion del cliente", 
                ylab="Numero promedio de contactos",
                col = c("blue","red"),
                cex.axis = 1,                              
                ylim = c(0,3.5
                         ))   # Densidad para relleno de la barras, por defecto 0
text(bar2, promedio2, labels=promedio2, pos=3, offset=0.8)


# Para el Ingreso
promedio3 <-  round(tapply(balance,y,mean),2)

bar3 <- barplot(promedio3,
                main="Grafico de Barras: Balance del cliente segun suscripcion",
                xlab="Situacion del cliente", 
                ylab="Balance o CLV del cliente",
                col = c("blue","red"),
                cex.axis = 1,                              
                ylim = c(0,2000))                                                          # Densidad para relleno de la barras, por defecto 0
text(bar3, promedio3, labels=promedio3, pos=3, offset=0.8)

# Ejercicios
# Presente un grafico de la EDAD promedio vs SUSCRIPCION y usando barplot() del paquete graphics






#-----------------------------------------------------------
# Usando ggplot2

# Para la Edad
promedio1 <-  round(tapply(edad,y,mean),2) ; promedio1
promedio1.df <- as.data.frame(promedio1)
rownames(promedio1.df)
str(promedio1.df)

ggplot(promedio1.df, aes(x=rownames(promedio1.df), y=promedio1)) +
  geom_bar(stat='identity', fill=c("cadetblue","firebrick1")) +
  theme_light() +
  labs(title ="Edad promedio segun la Suscripcion del Cliente", 
       x="Suscripcion del Cliente", 
       y= "Edad Promedio") + 
  ylim(0,60) +
  geom_text(aes(label=promedio1), 
            vjust=1.5, 
            color="white") 


# Para el numero de Hijos
promedio2 <-  round(tapply(balance,y,mean),2) ; promedio2
promedio2.df <- as.data.frame(promedio2)
rownames(promedio2.df)
str(promedio2.df)

ggplot(promedio2.df, aes(x=rownames(promedio2.df), y=promedio2)) +
  geom_bar(stat='identity', fill=c("cadetblue","firebrick1")) +
  theme_light() +
  labs(title ="Balance o CLV promedio segun la Suscripcion del Cliente", 
       x="Suscripcion del Cliente", 
       y= "Balance o CLV promedio") + 
  ylim(0,1600) +
  geom_text(aes(label=promedio2), 
            vjust=1.5, 
            color="white") 


# Ejercicios
# Presente un grafico de DURACION o DIAS vs SUSCRIPCION usando ggplot2
# Use promedio y promedio.df



#---------------------------------------------------------------
# Diagrama de cajas (BoxPlot)

# Diagrama de Cajas de variables cuantitativas segun CHURN
options(scipen=999)
boxplot(edad ~ y, 
        main= "BoxPlot de EDAD  vs SUSCRIPCION",
        xlab = "Cluster", 
        col = c("red","blue"))

boxplot(balance ~ y, 
        main= "BoxPlot de INGRESO vs CHURN",
        xlab = "Cluster",
        ylim=c(0,100000),
        col = c("red","blue"))

# Ejercicios
# Presente un grafico de DURACION o DIAS vs SUSCRIPCION usando boxplot()



#-------------------------------------------------------------------------------
# CONSIDERACIONES FINALES

# Para ver la categoria de referencia
contrasts(datos$mora)
contrasts(datos$y)

# Para cambiar la categoria de referencia
datos$mora = relevel(datos$mora,ref="si")
contrasts(datos$mora)


#------------------------------------------------------
# PARTE 2: PREDICTIVA Y CONSTRUCCIÓN DE MODELOS
#------------------------------------------------------

# Particion Muestral #

# Cuando trabajamos problemas de clasificacion, deberiamos estratificar
# de acuerdo a la variable de destino.
library(caret)
pmuestral <- createDataPartition(datos$y,     
                                  p = .75, 
                                  list = FALSE)

# Creamos los datasets de Train y Test
training <- datos[pmuestral,]
testing <- datos[ - pmuestral,]

# Una vez que hemos particionado los datos, modelamos la informacion
# Regresión Logistica

reglogit=glm(y~balance+edad+
                    mora,data=training,
            family = binomial(link = "logit")) # logit - probit


##################################################################

# Predecimos con el modelo entrenado la data de prueba
probabilidad = predict(, newdata=,type="response")
# Vamos a evaluar nuestro algoritmo

# Debemos convertirlo a clase (Cutoff)


# Calcular la matriz de confusion

matriz=confusionMatrix(PRED,data.prueba$Loan_Status,positive = "1")
matriz

# Una vez validado guardo el modelo predictivo o algoritmo.
saveRDS(reglogit,"Reg_Logit.rds") # Guardar un modelo predictivo
# Utilizamos el modelo entrenado para puntuar o scorear la data de puntuar o scorear

#------------------------------------------------------
# PARTE 3: DESPLIEGUE Y PRODUCTIVO DE MODELOS
#------------------------------------------------------

# Deseamos replicar o implementar el modelo.
# Leemos el modelo predictivo.
Reg_Logit<- readRDS("Reg_Logit.rds")

# Leemos el dataset de nuevos leads
library(data.table)
datos_n <-fread("NuevosLeads.csv",
              header=T, 
              verbose =FALSE, 
              stringsAsFactors=TRUE,
              showProgress =TRUE)

# Ejercicio.
# FIN !!
################################################################################
###########------------- Machine Learning  ------------------###################
################################################################################

# Capacitador: André Omar Chávez Panduro
# email: andrecp38@gmail.com / andre.chavez@urp.edu.pe
# Tema: Aprendizaje No Supervisado
############################################################################

# Cargamos las librerias a utilizar

library("cluster") # Activar librerias
library("fpc")
library("mclust")
library("dbscan")
library("readxl")

# Preparar los datos: normalizar y estandarizar ? Es necesario ?

carros<-read.csv("BaseAutos.csv", header=TRUE, sep=",", dec=".")
View(carros) # Ver u observar el set de datos
summary(carros) # Resumen de la informacion
str(carros)

# Ojo: Los softwares leen los datos de una manera pre-configurada,
# para cambiar:

carros$ID <- as.factor(carros$ID)

# Observacion:
# La segmentacion siempre se hace con variables numericas,las categoricas
# se usan para perfilamiento.

numericos <- sapply(carros, is.numeric) 
# Le aplicamos el argumento2 a el argumento1.


# Me quedo solo con las numericas
carrosnum<- carros[ ,numericos]


# Necesitamos las variables de identificacion u otras variables de perfilamiento
carroslabel<-paste(carros$fabricante,carros$modelo)


# Estandarizamos: ( Fundamental )
# Los algoritmos de segmentacion necesitan estandarizar la data
head(carrosnum)
carrosnormal<-scale(carrosnum) # Estamos estadarizando las variables 
# cuantitativas
head(carrosnormal)

# Metricas de distancia: usar dist() o libreria philentropy
ejemploeuclid<-dist(carrosnormal[2:3,],method="euclidean")

carrosdist<-dist(carrosnormal,method="euclidean")

########################################################################################
####### Cluster Jerarquico: Aglomerativo hclust() o agnes() , diana() es divisive ######
carrosjerarq <- hclust(carrosdist,method="ward.D")

# Mostrarlo con etiquetas
windows()
plot(carrosjerarq, labels=carroslabel)


###################################################################
######## Aplicamos el cluster no jerarquico KMeans ################

set.seed(100) # Semilla aleatoria

# Aplicamos el algoritmo de k-means
carroskmcluster<-kmeans(carrosnormal,centers=4,iter.max=1000000)

carroskmcluster$iter # Ver numero de iteraciones

# Cluster de pertenencia
carroskmcluster$cluster

# Centroides o centros de gravedad
carroskmcluster$centers

# Tamaño de los clusters
carroskmcluster$size
par(mfrow=c(1,1))
windows()
clusplot(carros,carroskmcluster$cluster, color=TRUE)


# Podriamos usar un método mas elaborado como PAM o CLARA
set.seed(100)
carrosmedoid<-pam(carrosnormal,k=4,stand=FALSE)
windows()
clusplot(carrosmedoid)

# Ver los valores y casos del medoide
carrosmedoid$medoids
carrosmedoid$id.med # Tamaño
carrosmedoid$clusinfo 

# Resumen del Modelo de Medoides
summary(carrosmedoid)
#los valores para guardarlos
carrosmedoid$clustering


##############################################
# Seleccionar el numero optimo de clusters####
##############################################

# Calcula la suma total de cuadrados
wss <- (nrow(carrosnormal)-1)*sum(apply(carrosnormal,2,var))

# La calcula por clusters
for (i in 2:15) wss[i] <- sum(kmeans(carrosnormal,
                                     centers=i)$withinss)
# Codo de Yambu
windows()
plot(1:15, wss, type="b", xlab="Nummero de Clusters",
     ylab="Suma de cuadrados within") 

##############################################
#Metodos Avanzados de Eleccion de Clusters####
##############################################

# Evaluar usando el criterio CH Indice Calinski Harabasz
#es un indice basado/cercano a una F de anova
set.seed(123) 
clustering.ch <- kmeansruns(carrosnormal,krange=3:20,
                            criterion="asw",
                            iter.max=100, 
                            runs= 100,
                            critout=TRUE)
clustering.ch$bestk

# Evaluar usando el criterio ASW (average sillouethe width)
set.seed(2) #Para evitar aleatoriedad en los resultados
clustering.asw <- kmeansruns(carrosnormal,krange=2:10,criterion="asw",
                             iter.max=100, runs= 100,critout=TRUE)

clustering.asw$bestk
clustering.asw$crit

#Evaluar con gap statistic
#mira el minimo k tal que el gap sea mayor que el gap de k+1 restado de su desviacion
gscar<-clusGap(carrosnormal,FUN=kmeans,K.max=8,B=60)
gscar

#validar resultados- consistencia
kclusters <- clusterboot(carrosnormal,B=1000,
                         clustermethod=kmeansCBI,
                         k=4,seed=5)

#la validacion del resultado. <0.65 o .75 muy bueno; < 0.65 Bueno >=0.55;
# < 0.55 Bueno >=0.45 regular;

kclusters$bootmean
#el grupo de pertenencia
hgroups <- kclusters$result$partition

### FIN ####

import matplotlib.pyplot as plt

__errors__ = []

# Funcion para obtener la hipotesis de una serie de parametros (thetas) y de samples (valores de x)
def h(params, sample):
    # Variable para ir llevando la sumatoria del resultado de hipotesis
    acum = 0
    for i in range(len(params)):
        acum += params[i] * sample[i]
    return acum 

# Funcion gradient descent que nos ayuda a ajustar y minimzar la funcion de costo mediante la actualización de sus parametros.
def GD(params, samples, lr, y):
    # Extraemos de los parametros el bias. Este es importante agregarlo en nuestras funciones hipotesis para obtener el valor real de la misma
    # Lo extraemos debido a que en la funcion h me daba errores en la multiplicacion, porque teniamos 3 parametros que multiplicar por 2 'x's
    # entonces a la hora de multiplicar, me daba en un punto que la lista sample estaba 'out of bounds'
    bias = params[-1]
    # Creamos la variable de new_params para no tener conflictos con los indices en la multiplicacion de la funcion de hipotesis
    new_params = params[:-1]
    # Creamos lista del tamaño de de new_params
    temp = list(new_params)
    n = len(samples)
    # Hacemos la funcion gradiente descendiente por cada parametro que tenemos
    for j in range(len(new_params)):
        # Variable para llevar la sumatoria
        acum=0
        # Hacemos la sumatoria por cada x que tenemos
        for i in range(n):
            # Llamamos la funcion h con el parametro de samples[i] para sacar nuetra hipotesis y para ello necesitamos los valores de x1 y x2
            hyp = h(new_params, samples[i]) + bias
            #Error en prediccion. Comparamos la prediccion con el valor real y multiplicamos por la x correspondiente
            # Agarramos i j para fijar la columna con el valor de j, porque los valores de una columna representan los valores de la misma sample
            acum += (hyp-y[i]) * samples[i][j]
        # Agregamos a la lista el valor final de la funcion gradiente descendiente del parametro(theta) correspondiente
        temp[j] = params[j] -((lr/n) * acum)
        #print(temp[j])
    # Agregamos la funcion gradiente descendiente del bias a nuestra lista
    temp.append(GD_B(new_params, samples, lr, y, bias))
    return temp

# Funcion gradiente descendiente del bias 
# La hice una funcion separada porque la funcion del bias es otra que la de los otros parametros
# Es la misma logica solo que no se multiplica por ninguna x de samples
def GD_B(params,samples, lr, y, bias):
    n = len(samples)
    acum = 0
    for i in range(n):
        hyp = h(params, samples[i]) + bias
        acum += (hyp-y[i])
    res = bias - ((lr/n) * acum)
    #print(res)
    return res

# Funcion para obtener el MSE de nuestro modelo
# Basicamente lo que hace es obtener la hipotesis con nuestros parametros actuales y los samples para al final compararlo con el valor real
# y ver qué tan preciso es nuestro modelo
def MSE(params, samples, y):
     n = len(samples)
     error_accum = 0
     bias = params[-1]
     new_params = params[:-1]
     for i in range(n):
          hyp = h(new_params,samples[i]) + bias
          print( "hyp  %f  y %f " % (hyp,  y[i]))
          error_accum += (hyp - y[i]) ** 2
     mse = error_accum / n
     __errors__.append(mse)

# La normalizacion de los samples nos ayuda para que cada dimensión (feature) de nuestro data set tenga el mismo peso a la hora de optimizar nuestros diagnosticos. 
# Es decir, si tengo dos features, edad e ingreso, la columna de edad tendra un rango de 1-100 e ingreso tendra uno de 1000-1000000. Si no aplicamos
# normalizacion, obviamente la columna ingreso dominará la predicción y el resultado no nos dará una buena representación de la vida real. 
# Así que aplicamos la normalización para que nuestro gradient descent pueda converger mejor, sea menos probable que se quede atorado en una parte,
# y además nos da una mejor representación de la realidad
def normalize(samples):
    # Hacemos un transpose de los samples
    transposed_samples = list(map(list, zip(*samples)))
    
    for i in range(1, len(transposed_samples)):
        # OBtenemos el promedio de la columna y su maximo
        avg = sum(transposed_samples[i]) / len(transposed_samples[i])
        max_val = max(transposed_samples[i])
        
        # Normalizamos cada elemento de la columna
        for j in range(len(transposed_samples[i])):
            transposed_samples[i][j] = (transposed_samples[i][j] - avg) / max_val
    
    # Regresamos los samples normalizados
    normalized_samples = list(map(list, zip(*transposed_samples)))
    return normalized_samples

# Parametros originales
params = [0,0,0]
# Samples de x1 y x2
samples = [[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7]]
# Valores reales. Nuestro modelo intentara acercarse lo mas que pueda a estos valores.
y = [2,4,6,8,10,12,14]

# Learning rate. Esto define qué tan rapido queremos que vaya aprendiendo nuestro modelo
lr = 0.01

# Número de iteraciones
epochs = 0

# Normalizar los samples para ayudar al gradient descent a converger de manera más eficiente
samples = normalize(samples)


# Vamos a iterar las veces que sea necesario hasta que nuestro error sea menor a un valor determinado
while True:
    old_params = list(params)
    # Sacamos nuevos parametros de nuestros parametros actuales 
    params = GD(params,samples,lr,y)
    # Obtenemos el error de esos nuevos parametros
    MSE(params,samples,y)
    # Variable para checar cuál es el error más reciente de nuestro modelo
    # ¿Por qué el más reciente? Porque nuestro error debe disminuir siempre con cada iteracion, entonces siempre con una nueva iteración
    # tendremos un error más bajo
    error = __errors__[-1]
    print(error)
    # Aumentamos el numero de iteraciones
    epochs += 1
    # Si llegamos a nuestro objetivo, cerrar el ciclo
    if (old_params == params or error <= 0.01):
        print("samples: ", samples)
        print("final params: ", params)
        print(error)
        break


print(f"# of epochs: ",epochs)

plt.plot(__errors__)
plt.title('Min Square Error per iteration')
plt.xlabel('Iterations')
plt.ylabel('Value of Error')
plt.show()
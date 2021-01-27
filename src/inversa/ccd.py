#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Robótica Computacional - Curso 2020/2021
# Grado en Ingeniería Informática (Cuarto)
# Práctica: Resolución de la cinemática inversa mediante CCD
#           (Cyclic Coordinate Descent).

import sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
import colorsys as cs


# ******************************************************************************
# Declaración de funciones

def muestra_origenes(O, final=0):
    # Muestra los orígenes de coordenadas para cada articulación
    print('Origenes de coordenadas:')
    for i in range(len(O)):
        print('(O' + str(i) + ')0\t= ' + str([round(j, 3) for j in O[i]]))
    if final:
        print('E.Final = ' + str([round(j, 3) for j in final]))


def muestra_robot(O, obj):
    # Muestra el robot graficamente
    plt.figure(1)
    plt.xlim(-L, L)
    plt.ylim(-L, L)
    T = [np.array(o).T.tolist() for o in O]
    for i in range(len(T)):
        plt.plot(T[i][0], T[i][1], '-o', color=cs.hsv_to_rgb(i / float(len(T)), 1, 1))
    plt.plot(obj[0], obj[1], '*')
    plt.show()
    raw_input()
    plt.clf()


def matriz_T(d, th, a, al):
    # Calcula la matriz T (ángulos de entrada en grados)

    return [[cos(th), -sin(th) * cos(al), sin(th) * sin(al), a * cos(th)]
        , [sin(th), cos(th) * cos(al), -sin(al) * cos(th), a * sin(th)]
        , [0, sin(al), cos(al), d]
        , [0, 0, 0, 1]
            ]


def cin_dir(th, a):
    # Sea 'th' el vector de thetas
    # Sea 'a'  el vector de longitudes
    T = np.identity(4)
    o = [[0, 0]]
    for i in range(len(th)):
        T = np.dot(T, matriz_T(0, th[i], a[i], 0))
        tmp = np.dot(T, [0, 0, 0, 1])
        o.append([tmp[0], tmp[1]])
    return o


# ******************************************************************************
# Cálculo de la cinemática inversa de forma iterativa por el método CCD

# valores articulares arbitrarios para la cinemática directa inicial
th = [0., 0., 0.]
a = [5., 5., 5.]
L = sum(a) + 10  # variable para representación gráfica
EPSILON = .01

# Definir limitación en el movimiento de los ángulos
valoresMin = [-(3 * pi / 4), 0., -(3 * pi / 4)]
valoresMax = [(3 * pi / 4), 10., (3 * pi / 4)]

# Indicar si el tipo de articulación es prismática
tipo = [0, 1, 0]

# modo interactivo
plt.ion()

# introducción del punto para la cinemática inversa
if len(sys.argv) != 3:
    sys.exit("python " + sys.argv[0] + " x y")
objetivo = [float(i) for i in sys.argv[1:]]

O = range(len(th) + 1)  # Reservamos estructura en memoria
O[0] = cin_dir(th, a)  # Calculamos la posicion inicial
print("- Posicion inicial:")
muestra_origenes(O[0])

dist = float("inf")
prev = 0.
iteracion = 1
while dist > EPSILON and abs(prev - dist) > EPSILON / 100.:
    prev = dist
    # Para cada combinación de articulaciones:
    for i in range(len(th)):
        # cálculo de la cinemática inversa:
        print("i=", i, "\nobjetivo=", objetivo, "\nO[i][-1]=", O[i][-1], "\nO[i]=", O[i])

        # Si la articulación no es prismática:
        if tipo[len(th) - i - 1] == 0:

            # Definimos 2 vectores distancia para realizar cálculos:
            # Calculamos la distancia entre la articulación actual y la final
            vectorDistanciaIF = np.subtract(O[i][len(th)], O[i][len(th) - i - 1])

            # Calculamos la distancia entre la articulación actual y el objetivo
            vectorDistanciaIO = np.subtract(objetivo, O[i][len(th) - i - 1])

            # Teniendo las distancias, estas definen un paralelogramo
            # Se tienen 2 ángulos, uno conformado entre el eje X y el eje distancia al objetivo (a1)
            # El otro se conforma desde el eje actual hasta el eje X (a2)
            # Calculamos las 2 arcotangentes para obtener el incremento
            a2 = atan2(vectorDistanciaIO[1], vectorDistanciaIO[0])
            a1 = atan2(vectorDistanciaIF[1], vectorDistanciaIF[0])

            # Incrementamos el valor del ángulo theta
            th[len(th) - i - 1] += (a2 - a1)

            # Aseguramos que el valor obtenido está en el rango [-PI, PI]
            while th[len(th) - i - 1] < -pi:
                th[len(th) - i - 1] += 2 * pi
            while th[len(th) - i - 1] > pi:
                th[len(th) - i - 1] -= 2 * pi

            # Añadimos la limitación de los ángulos en las articulaciones rotacionales
            th[len(th) - i - 1] = max(valoresMin[len(th) - i - 1], th[len(th) - i - 1])
            th[len(th) - i - 1] = min(valoresMax[len(th) - i - 1], th[len(th) - i - 1])

        # Si la articulación es prismática
        if tipo[len(th) - i - 1] == 1:
            # Calcular TitaP
            # Se define como el sumatorio de los ángulos anteriores
            j = 0
            titaP = 0
            while j < len(th) - i:
                titaP += th[j]
                j += 1

            # Hallamos vector desplazamiento.
            # Será un vector unitario con la dirección de la variable prismática
            vectorDesplazamiento = [cos(titaP), sin(titaP)]

            # Vector de acercamiento
            # Este vector define la distancia entre la articulación final y el objetivo
            vectorAcercamiento = np.subtract(objetivo, O[i][len(th)])

            # Calcular proyección de un punto frente al otro usando el producto escalar (np.dot)
            # Aumentar la distancia con el valor obtenido al hallar la proyección
            a[len(th) - i - 1] += np.dot(vectorDesplazamiento, vectorAcercamiento)

            # Limitación de desplazamientos en articulación prismática
            a[len(th) - i - 1] = max(valoresMin[len(th) - i - 1], a[len(th) - i - 1])
            a[len(th) - i - 1] = min(valoresMax[len(th) - i - 1], a[len(th) - i - 1])

        O[i + 1] = cin_dir(th, a)

    dist = np.linalg.norm(np.subtract(objetivo, O[-1][-1]))
    print("\n- Iteracion " + str(iteracion) + ':')
    muestra_origenes(O[-1])
    muestra_robot(O, objetivo)
    print("Distancia al objetivo = " + str(round(dist, 5)))
    iteracion += 1
    O[0] = O[-1]

if dist <= EPSILON:
    print("\n" + str(iteracion) + " iteraciones para converger.")
else:
    print("\nNo hay convergencia tras " + str(iteracion) + " iteraciones.")
print("- Umbral de convergencia epsilon: " + str(EPSILON))
print("- Distancia al objetivo:          " + str(round(dist, 5)))
print("- Valores finales de las articulaciones:")
for i in range(len(th)):
    print("  theta" + str(i + 1) + " = " + str(round(th[i], 3)))
for i in range(len(th)):
    print("  L" + str(i + 1) + "     = " + str(round(a[i], 3)))

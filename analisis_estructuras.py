"""
Análisis de Estructura a Gran Escala del Universo
Estudiantes: Alejandro Guerrero - 202510898
Ariana Lozano - 202410609
Samara Martinez - 202221057
Sofia Vasquez - 202123910
"""

import numpy as np
from astropy.table import Table, Column
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# PARÁMETROS DEL ANÁLISIS
# -----------------------
# N_VECINOS: número de galaxias vecinas que se usan para estimar la densidad local
N_VECINOS = 20
# DISTANCIA_AGRUPACION_CLUSTER: distancia máxima para considerar que dos galaxias
# de tipo CÚMULO pertenecen al mismo grupo (algoritmo Friends-of-Friends)
DISTANCIA_AGRUPACION_CLUSTER = 50.0  # millones años luz
# DISTANCIA_AGRUPACION_VOID: distancia máxima para agrupar galaxias de tipo VACÍO
DISTANCIA_AGRUPACION_VOID = 80.0     # millones años luz
# DISTANCIA_MAXIMA_FILAMENTO: separación máxima entre centros de cúmulos
# para trazar un filamento entre ellos
DISTANCIA_MAXIMA_FILAMENTO = 300.0   # millones años luz
# PASO_MUESTREO_FILAMENTO: separación entre puntos muestreados a lo largo
# del filamento (recta que une dos cúmulos)
PASO_MUESTREO_FILAMENTO = 10.0       # millones años luz

# Variables "globales" para compartir información entre funciones
# (se van llenando a lo largo del flujo principal)
datos_galaxias = None          # Tabla con todas las galaxias leídas del archivo
tabla_clusters = None          # Tabla con los cúmulos identificados
tabla_voids = None             # Tabla con los vacíos identificados
tabla_filamentos = None        # Tabla con los puntos que trazan los filamentos


# --------------------------------------------------------------------
# Utilidades internas
# --------------------------------------------------------------------

def _coords_from_table(tabla, cols=('X', 'Y', 'Z')):
    """
    Devuelve un array (N, 3) con las coordenadas de la tabla.

    Parámetros
    ----------
    tabla : astropy.table.Table
        Tabla que contiene las columnas de coordenadas.
    cols : tupla de str
        Nombres de las columnas que contienen X, Y, Z (en ese orden).

    Retorna
    -------
    coords : ndarray de forma (N, 3)
        Matriz con las posiciones (X, Y, Z) de cada galaxia.
    """
    return np.vstack([tabla[cols[0]], tabla[cols[1]], tabla[cols[2]]]).T


def _friends_of_friends(coords, max_dist):
    """
    Algoritmo sencillo de Friends-of-Friends (FoF) para agrupar puntos.

    Dos puntos pertenecen al mismo grupo si se pueden conectar por una cadena
    de "amigos" donde cada par está a una distancia menor que max_dist.

    Parámetros
    ----------
    coords : ndarray (N, 3)
        Coordenadas cartesianas de los puntos.
    max_dist : float
        Distancia de enlace (linking length).

    Retorna
    -------
    grupos : lista de arrays de enteros
        Cada elemento es un array con los índices de coords que pertenecen
        a un mismo grupo.
    """
    n = coords.shape[0]
    visited = np.zeros(n, dtype=bool)  # Marca si ya fue asignado a un grupo
    grupos = []
    max_dist2 = max_dist * max_dist    # Trabajamos con distancias al cuadrado

    for i in range(n):
        if visited[i]:
            # Si ya está en un grupo, lo saltamos
            continue

        # Nuevo grupo iniciando desde el punto i
        stack = [i]        # pila para hacer un recorrido tipo DFS
        visited[i] = True
        miembros = []      # almacenará los índices del grupo actual

        while stack:
            j = stack.pop()
            miembros.append(j)

            # Distancias desde el punto j a todos los puntos
            diff = coords - coords[j]
            dist2 = np.einsum('ij,ij->i', diff, diff)

            # Vecinos que están dentro de la distancia de enlace y no visitados
            vecinos = np.where((dist2 <= max_dist2) & (~visited))[0]

            if vecinos.size > 0:
                visited[vecinos] = True
                # Agregamos vecinos a la pila para seguir explorando
                stack.extend(vecinos.tolist())

        # Guardamos el grupo encontrado
        grupos.append(np.array(miembros, dtype=int))

    return grupos


# --------------------------------------------------------------------
# Paso 0: Lectura de datos
# --------------------------------------------------------------------

def leer_datos():
    """
    Leer el archivo galaxy_cartesian_coordinates.ecsv y guardar en datos_galaxias.

    Se asume que el archivo se encuentra en el mismo directorio que este script.
    """
    global datos_galaxias
    # Leemos usando el formato ECSV con astropy
    datos_galaxias = Table.read('galaxy_cartesian_coordinates.ecsv', format='ascii.ecsv')
    return datos_galaxias


# --------------------------------------------------------------------
# Paso 1: Densidad local
# --------------------------------------------------------------------

def calcular_densidad_local():
    """
    Calcular la densidad local de cada galaxia usando los N_VECINOS más cercanos.

    Para cada galaxia:
      - Se buscan los N_VECINOS más cercanos.
      - Se define una esfera cuyo radio es la distancia al vecino más lejano
        de esos N_VECINOS.
      - La densidad local es N_VECINOS / volumen_de_la_esfera.

    El resultado se guarda en la columna 'DENSIDAD_LOCAL' de datos_galaxias.
    """
    global datos_galaxias
    if datos_galaxias is None:
        raise RuntimeError("Los datos no han sido leídos. Llama a leer_datos() primero.")

    coords = _coords_from_table(datos_galaxias)  # (N, 3)
    n = coords.shape[0]
    k = N_VECINOS

    densidades = np.empty(n, dtype=float)

    for i in range(n):
        # Distancias (al cuadrado) a todas las galaxias desde la galaxia i
        diff = coords - coords[i]
        dist2 = np.einsum('ij,ij->i', diff, diff)

        # Ignoramos la distancia a sí mismo (sería 0)
        dist2[i] = np.inf

        # Índices de los k vecinos más cercanos usando argpartition (más eficiente que sort)
        idx_k = np.argpartition(dist2, k)[:k]
        # El radio es la raíz cuadrada de la distancia al vecino más lejano dentro de esos k
        r_k = np.sqrt(np.max(dist2[idx_k]))

        if r_k > 0.0:
            # Volumen de una esfera de radio r_k
            volumen = (4.0 / 3.0) * np.pi * r_k**3
            # Densidad = número de vecinos / volumen de la esfera
            densidades[i] = k / volumen
        else:
            # Caso degenerado: si el radio es 0, ponemos densidad 0
            densidades[i] = 0.0

    # Añadimos la columna de densidad local a la tabla de galaxias
    datos_galaxias['DENSIDAD_LOCAL'] = densidades


def clasificar_galaxias():
    """
    Clasificar galaxias en CUMULO, FILAMENTO, VACIO según umbrales
    de densidad local relativos a la mediana.

    Regla:
      - CUMULO: densidad > 2 × densidad_mediana
      - FILAMENTO: 0.5 × densidad_mediana ≤ densidad ≤ 2 × densidad_mediana
      - VACIO: densidad < 0.5 × densidad_mediana
    """
    global datos_galaxias
    if datos_galaxias is None or 'DENSIDAD_LOCAL' not in datos_galaxias.colnames:
        raise RuntimeError("Primero hay que calcular la densidad local.")

    dens = datos_galaxias['DENSIDAD_LOCAL']

    # Usamos la mediana de las densidades > 0 para evitar efectos de ceros raros
    dens_pos = dens[dens > 0]
    if len(dens_pos) == 0:
        dens_mediana = np.median(dens)
    else:
        dens_mediana = np.median(dens_pos)

    # Creamos un array de tipos de galaxia, inicialmente FILAMENTO
    tipos = np.empty(len(dens), dtype='U10')
    tipos[:] = 'FILAMENTO'

    # Aplicamos los umbrales
    tipos[dens > 2.0 * dens_mediana] = 'CUMULO'
    tipos[dens < 0.5 * dens_mediana] = 'VACIO'

    # Guardamos la clasificación y la densidad mediana en los metadatos
    datos_galaxias['CLASE'] = tipos
    datos_galaxias.meta['DENSIDAD_MEDIANA'] = float(dens_mediana)


# --------------------------------------------------------------------
# Paso 2: Identificar cúmulos
# --------------------------------------------------------------------

def identificar_clusters():
    """
    Identificar cúmulos a partir de las galaxias clasificadas como CUMULO.

    Procedimiento:
      - Se seleccionan solo las galaxias con CLASE == 'CUMULO'.
      - Se agrupan con Friends-of-Friends usando DISTANCIA_AGRUPACION_CLUSTER.
      - Para cada grupo se calcula:
          * Centro: promedio de las coordenadas (X, Y, Z).
          * Radio: máxima distancia desde el centro a cualquiera de las galaxias.
          * N_GALAXIAS: número de galaxias en el cúmulo.

    El resultado se guarda en la tabla global tabla_clusters y también se retorna.
    """
    global datos_galaxias, tabla_clusters
    if datos_galaxias is None or 'CLASE' not in datos_galaxias.colnames:
        raise RuntimeError("Primero hay que clasificar las galaxias.")

    # Máscara de galaxias clasificadas como CÚMULO
    mask_c = (datos_galaxias['CLASE'] == 'CUMULO')
    if np.sum(mask_c) == 0:
        # No hay cúmulos, devolvemos una tabla vacía con las columnas correctas
        tabla_clusters = Table()
        tabla_clusters['CLUSTER_ID'] = Column([], dtype=int)
        tabla_clusters['X_CENTRO'] = Column([], dtype=float)
        tabla_clusters['Y_CENTRO'] = Column([], dtype=float)
        tabla_clusters['Z_CENTRO'] = Column([], dtype=float)
        tabla_clusters['RADIO'] = Column([], dtype=float)
        tabla_clusters['N_GALAXIAS'] = Column([], dtype=int)
        return tabla_clusters

    # Subtabla solo con galaxias de cúmulo
    sub = datos_galaxias[mask_c]
    coords_c = _coords_from_table(sub)

    # Agrupamos con FoF
    grupos = _friends_of_friends(coords_c, DISTANCIA_AGRUPACION_CLUSTER)

    cluster_ids = []
    x_centro = []
    y_centro = []
    z_centro = []
    radios = []
    n_gal = []

    # Recorremos cada grupo de cúmulo
    for cid, indices in enumerate(grupos, start=1):
        puntos = coords_c[indices]
        # Centro del cúmulo: promedio de las posiciones
        centro = np.mean(puntos, axis=0)
        # Distancias desde el centro a cada galaxia del grupo
        distancias = np.sqrt(np.sum((puntos - centro)**2, axis=1))
        # Radio: la distancia máxima
        radio = np.max(distancias) if len(distancias) > 0 else 0.0

        cluster_ids.append(cid)
        x_centro.append(centro[0])
        y_centro.append(centro[1])
        z_centro.append(centro[2])
        radios.append(radio)
        n_gal.append(len(indices))

    # Construimos la tabla de cúmulos
    tabla_clusters = Table()
    tabla_clusters['CLUSTER_ID'] = np.array(cluster_ids, dtype=int)
    tabla_clusters['X_CENTRO'] = np.array(x_centro, dtype=float)
    tabla_clusters['Y_CENTRO'] = np.array(y_centro, dtype=float)
    tabla_clusters['Z_CENTRO'] = np.array(z_centro, dtype=float)
    tabla_clusters['RADIO'] = np.array(radios, dtype=float)
    tabla_clusters['N_GALAXIAS'] = np.array(n_gal, dtype=int)

    return tabla_clusters


# --------------------------------------------------------------------
# Paso 3: Identificar vacíos
# --------------------------------------------------------------------

def identificar_voids():
    """
    Identificar vacíos agrupando galaxias clasificadas como VACIO.

    Nota: En la realidad, un vacío es una región "sin" galaxias,
    pero como solo tenemos galaxias, usamos las galaxias de baja densidad
    como trazadores de las regiones vacías.

    Procedimiento:
      - Se seleccionan galaxias con CLASE == 'VACIO'.
      - Se agrupan con Friends-of-Friends usando DISTANCIA_AGRUPACION_VOID.
      - Para cada grupo se calcula un centro y un radio efectivo (máxima
        distancia al centro).

    El resultado se guarda en tabla_voids.
    """
    global datos_galaxias, tabla_voids
    if datos_galaxias is None or 'CLASE' not in datos_galaxias.colnames:
        raise RuntimeError("Primero hay que clasificar las galaxias.")

    # Máscara de galaxias de vacío
    mask_v = (datos_galaxias['CLASE'] == 'VACIO')
    if np.sum(mask_v) == 0:
        # Si no hay galaxias de vacío, tabla vacía
        tabla_voids = Table()
        tabla_voids['VOID_ID'] = Column([], dtype=int)
        tabla_voids['X_CENTRO'] = Column([], dtype=float)
        tabla_voids['Y_CENTRO'] = Column([], dtype=float)
        tabla_voids['Z_CENTRO'] = Column([], dtype=float)
        tabla_voids['RADIO'] = Column([], dtype=float)
        return tabla_voids

    # Subtabla solo con galaxias en vacíos
    sub = datos_galaxias[mask_v]
    coords_v = _coords_from_table(sub)

    # Agrupamos con FoF
    grupos = _friends_of_friends(coords_v, DISTANCIA_AGRUPACION_VOID)

    void_ids = []
    x_centro = []
    y_centro = []
    z_centro = []
    radios = []

    for vid, indices in enumerate(grupos, start=1):
        puntos = coords_v[indices]
        centro = np.mean(puntos, axis=0)
        distancias = np.sqrt(np.sum((puntos - centro)**2, axis=1))
        radio = np.max(distancias) if len(distancias) > 0 else 0.0

        void_ids.append(vid)
        x_centro.append(centro[0])
        y_centro.append(centro[1])
        z_centro.append(centro[2])
        radios.append(radio)

    # Construimos la tabla de vacíos
    tabla_voids = Table()
    tabla_voids['VOID_ID'] = np.array(void_ids, dtype=int)
    tabla_voids['X_CENTRO'] = np.array(x_centro, dtype=float)
    tabla_voids['Y_CENTRO'] = np.array(y_centro, dtype=float)
    tabla_voids['Z_CENTRO'] = np.array(z_centro, dtype=float)
    tabla_voids['RADIO'] = np.array(radios, dtype=float)

    return tabla_voids


# --------------------------------------------------------------------
# Paso 4: Trazar filamentos
# --------------------------------------------------------------------

def trazar_filamentos():
    """
    Trazar filamentos entre cúmulos cercanos.

    Para cada par de cúmulos:
      - Se calcula la distancia entre sus centros.
      - Si la distancia es menor que DISTANCIA_MAXIMA_FILAMENTO:
           * Se genera una serie de puntos igualmente espaciados a lo
             largo del segmento que une ambos centros (interpolación lineal).
           * Cada conjunto de puntos se etiqueta con un FILAMENTO_ID.

    El resultado se guarda en tabla_filamentos.
    """
    global tabla_clusters, tabla_filamentos
    if tabla_clusters is None:
        raise RuntimeError("Primero hay que identificar los cúmulos.")

    n_cl = len(tabla_clusters)
    if n_cl == 0:
        # No hay cúmulos, por lo tanto no hay filamentos
        tabla_filamentos = Table()
        tabla_filamentos['FILAMENTO_ID'] = Column([], dtype=int)
        tabla_filamentos['X'] = Column([], dtype=float)
        tabla_filamentos['Y'] = Column([], dtype=float)
        tabla_filamentos['Z'] = Column([], dtype=float)
        tabla_filamentos['CLUSTER_ORIGEN'] = Column([], dtype=int)
        tabla_filamentos['CLUSTER_DESTINO'] = Column([], dtype=int)
        return tabla_filamentos

    # Coordenadas de los centros de los cúmulos
    centros = _coords_from_table(tabla_clusters, cols=('X_CENTRO', 'Y_CENTRO', 'Z_CENTRO'))

    filas_fil = []  # aquí acumulamos las filas (puntos) de todos los filamentos
    fil_id = 0      # contador de filamentos

    # Recorremos todos los pares de cúmulos (i < j para no repetir)
    for i in range(n_cl):
        for j in range(i + 1, n_cl):
            p_i = centros[i]
            p_j = centros[j]
            diff = p_j - p_i
            dist = np.sqrt(np.dot(diff, diff))  # distancia entre centros

            # Verificamos si vale la pena trazar un filamento
            if dist <= DISTANCIA_MAXIMA_FILAMENTO and dist > 0.0:
                fil_id += 1
                # Número de puntos a lo largo del filamento, según el paso
                n_steps = int(dist // PASO_MUESTREO_FILAMENTO) + 2
                # Parámetro t entre 0 y 1 para interpolar a lo largo de la recta
                t = np.linspace(0.0, 1.0, n_steps)
                puntos = p_i[None, :] + t[:, None] * diff[None, :]
                for p in puntos:
                    filas_fil.append(
                        (fil_id, p[0], p[1], p[2],
                         int(tabla_clusters['CLUSTER_ID'][i]),
                         int(tabla_clusters['CLUSTER_ID'][j]))
                    )

    # Si no se generó ningún filamento, devolvemos una tabla vacía
    if len(filas_fil) == 0:
        tabla_filamentos = Table()
        tabla_filamentos['FILAMENTO_ID'] = Column([], dtype=int)
        tabla_filamentos['X'] = Column([], dtype=float)
        tabla_filamentos['Y'] = Column([], dtype=float)
        tabla_filamentos['Z'] = Column([], dtype=float)
        tabla_filamentos['CLUSTER_ORIGEN'] = Column([], dtype=int)
        tabla_filamentos['CLUSTER_DESTINO'] = Column([], dtype=int)
        return tabla_filamentos

    # Convertimos la lista de filas a una estructura numpy y luego a tabla
    filas_fil = np.array(filas_fil, dtype=[('FILAMENTO_ID', int),
                                           ('X', float),
                                           ('Y', float),
                                           ('Z', float),
                                           ('CLUSTER_ORIGEN', int),
                                           ('CLUSTER_DESTINO', int)])

    tabla_filamentos = Table(filas_fil)

    return tabla_filamentos


# --------------------------------------------------------------------
# Gráficas
# --------------------------------------------------------------------

def generar_graficas():
    """
    Generar los 4 PDFs requeridos:

    1) distribucion_galaxias.pdf
       - Proyecciones XY, XZ, YZ de todas las galaxias.

    2) estructuras_clasificadas.pdf
       - Misma distribución pero coloreando por CLASE (VACIO, FILAMENTO, CUMULO).

    3) clusters_y_voids.pdf
       - Muestra los centros y radios (círculos) de cúmulos y vacíos.

    4) red_cosmica_completa.pdf
       - Galaxias en gris + filamentos en azul + cúmulos en rojo + vacíos en verde.
    """
    global datos_galaxias, tabla_clusters, tabla_voids, tabla_filamentos
    if datos_galaxias is None:
        raise RuntimeError("No hay datos de galaxias.")

    # Extraemos las coordenadas básicas de las galaxias
    x = datos_galaxias['X']
    y = datos_galaxias['Y']
    z = datos_galaxias['Z']

    # ----------------------------------------------------------------
    # 1. distribucion_galaxias.pdf
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(6, 12))

    # Proyección XY
    axes[0].scatter(x, y, s=1, alpha=0.5)
    axes[0].set_xlabel('X [Mly]')
    axes[0].set_ylabel('Y [Mly]')
    axes[0].set_title('Proyección XY')

    # Proyección XZ
    axes[1].scatter(x, z, s=1, alpha=0.5)
    axes[1].set_xlabel('X [Mly]')
    axes[1].set_ylabel('Z [Mly]')
    axes[1].set_title('Proyección XZ')

    # Proyección YZ
    axes[2].scatter(y, z, s=1, alpha=0.5)
    axes[2].set_xlabel('Y [Mly]')
    axes[2].set_ylabel('Z [Mly]')
    axes[2].set_title('Proyección YZ')

    plt.tight_layout()
    plt.savefig('distribucion_galaxias.pdf')
    plt.close(fig)

    # ----------------------------------------------------------------
    # 2. estructuras_clasificadas.pdf
    # ----------------------------------------------------------------
    clases = datos_galaxias['CLASE']
    fig, axes = plt.subplots(3, 1, figsize=(6, 12))

    # Recorremos las tres proyecciones y coloreamos por CLASE
    for ax, (a, b, label_a, label_b) in zip(
            axes,
            [(x, y, 'X', 'Y'),
             (x, z, 'X', 'Z'),
             (y, z, 'Y', 'Z')]):

        # VACIO: verde
        m_v = (clases == 'VACIO')
        ax.scatter(a[m_v], b[m_v], s=2, c='g', alpha=0.5, label='VACIO')

        # FILAMENTO: azul
        m_f = (clases == 'FILAMENTO')
        ax.scatter(a[m_f], b[m_f], s=2, c='b', alpha=0.5, label='FILAMENTO')

        # CUMULO: rojo
        m_c = (clases == 'CUMULO')
        ax.scatter(a[m_c], b[m_c], s=2, c='r', alpha=0.5, label='CUMULO')

        ax.set_xlabel(f'{label_a} [Mly]')
        ax.set_ylabel(f'{label_b} [Mly]')
        ax.set_title(f'Proyección {label_a}{label_b}')
        ax.legend(markerscale=4, fontsize='small')

    plt.tight_layout()
    plt.savefig('estructuras_clasificadas.pdf')
    plt.close(fig)

    # ----------------------------------------------------------------
    # 3. clusters_y_voids.pdf
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(6, 12))

    # Coordenadas de clusters y voids (si existen)
    if tabla_clusters is not None and len(tabla_clusters) > 0:
        xc = tabla_clusters['X_CENTRO']
        yc = tabla_clusters['Y_CENTRO']
        zc = tabla_clusters['Z_CENTRO']
        rc = tabla_clusters['RADIO']
    else:
        xc = yc = zc = rc = np.array([])

    if tabla_voids is not None and len(tabla_voids) > 0:
        xv = tabla_voids['X_CENTRO']
        yv = tabla_voids['Y_CENTRO']
        zv = tabla_voids['Z_CENTRO']
        rv = tabla_voids['RADIO']
    else:
        xv = yv = zv = rv = np.array([])

    # Definimos las tres proyecciones para dibujar círculos de radio
    proyecciones = [
        ('XY', (xc, yc, rc), (xv, yv, rv), 'X [Mly]', 'Y [Mly]'),
        ('XZ', (xc, zc, rc), (xv, zv, rv), 'X [Mly]', 'Z [Mly]'),
        ('YZ', (yc, zc, rc), (yv, zv, rv), 'Y [Mly]', 'Z [Mly]')
    ]

    for ax, (nombre, cl, vo, label_a, label_b) in zip(axes, proyecciones):
        # Centros de cúmulos (círculos rojos)
        if len(cl[0]) > 0:
            ax.scatter(cl[0], cl[1], c='r', s=20, label='Centros Cúmulos')
            # Círculos que representan el radio del cúmulo
            for cx, cy, r in zip(cl[0], cl[1], cl[2]):
                circ = Circle((cx, cy), r, edgecolor='r', facecolor='none',
                              linestyle='--', linewidth=0.5, alpha=0.7)
                ax.add_patch(circ)

        # Centros de vacíos (círculos verdes)
        if len(vo[0]) > 0:
            ax.scatter(vo[0], vo[1], c='g', s=20, label='Centros Vacíos')
            for vx, vy, r in zip(vo[0], vo[1], vo[2]):
                circ = Circle((vx, vy), r, edgecolor='g', facecolor='none',
                              linestyle='--', linewidth=0.5, alpha=0.7)
                ax.add_patch(circ)

        ax.set_xlabel(label_a)
        ax.set_ylabel(label_b)
        ax.set_title(f'Cúmulos y Vacíos - Proyección {nombre}')
        ax.legend(fontsize='small')

    plt.tight_layout()
    plt.savefig('clusters_y_voids.pdf')
    plt.close(fig)

    # ----------------------------------------------------------------
    # 4. red_cosmica_completa.pdf
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(6, 12))

    for ax, (a, b, label_a, label_b) in zip(
            axes,
            [(x, y, 'X', 'Y'),
             (x, z, 'X', 'Z'),
             (y, z, 'Y', 'Z')]):

        # Galaxias en gris de fondo
        ax.scatter(a, b, s=1, c='0.7', alpha=0.5, label='Galaxias')

        # Filamentos (si existen): líneas azules
        if tabla_filamentos is not None and len(tabla_filamentos) > 0:
            ids = np.unique(tabla_filamentos['FILAMENTO_ID'])
            for fid in ids:
                m = (tabla_filamentos['FILAMENTO_ID'] == fid)
                xf = tabla_filamentos['X'][m]
                yf = tabla_filamentos['Y'][m]
                zf = tabla_filamentos['Z'][m]

                # Dependiendo de la proyección, usamos las coordenadas apropiadas
                if label_a == 'X' and label_b == 'Y':
                    ax.plot(xf, yf, '-', linewidth=0.7, alpha=0.7, c='b')
                elif label_a == 'X' and label_b == 'Z':
                    ax.plot(xf, zf, '-', linewidth=0.7, alpha=0.7, c='b')
                elif label_a == 'Y' and label_b == 'Z':
                    ax.plot(yf, zf, '-', linewidth=0.7, alpha=0.7, c='b')

        # Cúmulos (puntos rojos)
        if tabla_clusters is not None and len(tabla_clusters) > 0:
            if label_a == 'X' and label_b == 'Y':
                ax.scatter(xc, yc, c='r', s=20, label='Cúmulos')
            elif label_a == 'X' and label_b == 'Z':
                ax.scatter(xc, zc, c='r', s=20, label='Cúmulos')
            elif label_a == 'Y' and label_b == 'Z':
                ax.scatter(yc, zc, c='r', s=20, label='Cúmulos')

        # Vacíos (puntos verdes)
        if tabla_voids is not None and len(tabla_voids) > 0:
            if label_a == 'X' and label_b == 'Y':
                ax.scatter(xv, yv, c='g', s=20, label='Vacíos')
            elif label_a == 'X' and label_b == 'Z':
                ax.scatter(xv, zv, c='g', s=20, label='Vacíos')
            elif label_a == 'Y' and label_b == 'Z':
                ax.scatter(yv, zv, c='g', s=20, label='Vacíos')

        ax.set_xlabel(f'{label_a} [Mly]')
        ax.set_ylabel(f'{label_b} [Mly]')
        ax.set_title(f'Red Cósmica Completa - Proyección {label_a}{label_b}')
        ax.legend(fontsize='x-small')

    plt.tight_layout()
    plt.savefig('red_cosmica_completa.pdf')
    plt.close(fig)


# --------------------------------------------------------------------
# Escritura de catálogos y estadísticas
# --------------------------------------------------------------------

def escribir_catalogos():
    """
    Guardar en disco:

    - catalogo_clusters.ecsv       (cúmulos)
    - catalogo_voids.ecsv         (vacíos)
    - trazado_filamentos.ecsv     (puntos de filamentos)
    - estadisticas.txt            (resumen numérico del análisis)
    """
    global datos_galaxias, tabla_clusters, tabla_voids, tabla_filamentos

    # Guardar catálogos ECSV si existen las tablas correspondientes
    if tabla_clusters is not None:
        tabla_clusters.write('catalogo_clusters.ecsv',
                             format='ascii.ecsv', overwrite=True)

    if tabla_voids is not None:
        tabla_voids.write('catalogo_voids.ecsv',
                          format='ascii.ecsv', overwrite=True)

    if tabla_filamentos is not None:
        tabla_filamentos.write('trazado_filamentos.ecsv',
                               format='ascii.ecsv', overwrite=True)

    # ------------------------------------------------------------
    # Cálculo de estadísticas globales para el archivo de texto
    # ------------------------------------------------------------
    n_total = len(datos_galaxias) if datos_galaxias is not None else 0

    if datos_galaxias is not None and 'CLASE' in datos_galaxias.colnames:
        clases = datos_galaxias['CLASE']
        n_c = np.sum(clases == 'CUMULO')
        n_f = np.sum(clases == 'FILAMENTO')
        n_v = np.sum(clases == 'VACIO')
    else:
        n_c = n_f = n_v = 0

    def pct(n):
        """Porcentaje de n respecto al total de galaxias."""
        return 100.0 * n / n_total if n_total > 0 else 0.0

    # --- Estadísticas de cúmulos ---
    if tabla_clusters is not None and len(tabla_clusters) > 0:
        n_clusters = len(tabla_clusters)
        radio_prom_clusters = float(np.mean(tabla_clusters['RADIO']))
        idx_max_c = int(np.argmax(tabla_clusters['RADIO']))
        id_max_cluster = int(tabla_clusters['CLUSTER_ID'][idx_max_c])
        n_gal_max_cluster = int(tabla_clusters['N_GALAXIAS'][idx_max_c])
        radio_max_cluster = float(tabla_clusters['RADIO'][idx_max_c])
    else:
        n_clusters = 0
        radio_prom_clusters = 0.0
        id_max_cluster = -1
        n_gal_max_cluster = 0
        radio_max_cluster = 0.0

    # --- Estadísticas de vacíos ---
    if tabla_voids is not None and len(tabla_voids) > 0:
        n_voids = len(tabla_voids)
        radio_prom_voids = float(np.mean(tabla_voids['RADIO']))
        idx_max_v = int(np.argmax(tabla_voids['RADIO']))
        id_max_void = int(tabla_voids['VOID_ID'][idx_max_v])
        radio_max_void = float(tabla_voids['RADIO'][idx_max_v])
    else:
        n_voids = 0
        radio_prom_voids = 0.0
        id_max_void = -1
        radio_max_void = 0.0

    # --- Estadísticas de filamentos ---
    if tabla_filamentos is not None and len(tabla_filamentos) > 0:
        ids_fil = np.unique(tabla_filamentos['FILAMENTO_ID'])
        n_filamentos = len(ids_fil)

        longitudes = []
        for fid in ids_fil:
            m = (tabla_filamentos['FILAMENTO_ID'] == fid)
            x_f = np.array(tabla_filamentos['X'][m])
            y_f = np.array(tabla_filamentos['Y'][m])
            z_f = np.array(tabla_filamentos['Z'][m])
            # Aproximamos la longitud como la distancia entre el primer y el último punto
            dx = x_f[-1] - x_f[0]
            dy = y_f[-1] - y_f[0]
            dz = z_f[-1] - z_f[0]
            longitudes.append(np.sqrt(dx*dx + dy*dy + dz*dz))

        longitudes = np.array(longitudes)
        longitud_prom_filamentos = float(np.mean(longitudes)) if len(longitudes) > 0 else 0.0
        total_puntos_filamentos = len(tabla_filamentos)
    else:
        n_filamentos = 0
        longitud_prom_filamentos = 0.0
        total_puntos_filamentos = 0

    # ------------------------------------------------------------
    # Escritura del archivo de texto con las estadísticas
    # ------------------------------------------------------------
    with open('estadisticas.txt', 'w') as f:
        f.write("========================================\n")
        f.write("ESTRUCTURA A GRAN ESCALA - ESTADÍSTICAS\n")
        f.write("========================================\n\n")

        f.write(f"Número total de galaxias: {n_total}\n\n")

        f.write("CLASIFICACIÓN POR DENSIDAD\n")
        f.write("---------------------------\n")
        f.write(f"Galaxias en CUMULOS: {n_c} ({pct(n_c):.1f}%)\n")
        f.write(f"Galaxias en FILAMENTOS: {n_f} ({pct(n_f):.1f}%)\n")
        f.write(f"Galaxias en VACIOS: {n_v} ({pct(n_v):.1f}%)\n\n")

        f.write("CÚMULOS IDENTIFICADOS\n")
        f.write("---------------------\n")
        f.write(f"Número total de cúmulos: {n_clusters}\n")
        f.write(f"Radio promedio: {radio_prom_clusters:.2f} millones años luz\n")
        if n_clusters > 0:
            f.write(f"Cúmulo más grande: ID {id_max_cluster} "
                    f"({n_gal_max_cluster} galaxias, radio {radio_max_cluster:.2f})\n")
        f.write("\n")

        f.write("VACÍOS IDENTIFICADOS\n")
        f.write("--------------------\n")
        f.write(f"Número total de vacíos: {n_voids}\n")
        f.write(f"Radio promedio: {radio_prom_voids:.2f} millones años luz\n")
        if n_voids > 0:
            f.write(f"Vacío más grande: ID {id_max_void} "
                    f"(radio {radio_max_void:.2f} millones años luz)\n")
        f.write("\n")

        f.write("FILAMENTOS TRAZADOS\n")
        f.write("-------------------\n")
        f.write(f"Número de filamentos: {n_filamentos}\n")
        f.write(f"Longitud promedio: {longitud_prom_filamentos:.2f} millones años luz\n")
        f.write(f"Total de puntos trazados: {total_puntos_filamentos}\n")


# --------------------------------------------------------------------
# Programa principal
# --------------------------------------------------------------------

if __name__ == '__main__':
    # Flujo completo del análisis:
    leer_datos()             # 1) Leer catálogo de galaxias
    calcular_densidad_local()# 2) Calcular densidad local de cada galaxia
    clasificar_galaxias()    # 3) Clasificar en CÚMULO / FILAMENTO / VACÍO
    identificar_clusters()   # 4) Identificar cúmulos
    identificar_voids()      # 5) Identificar vacíos
    trazar_filamentos()      # 6) Trazar filamentos entre cúmulos cercanos
    generar_graficas()       # 7) Generar PDFs con las proyecciones
    escribir_catalogos()     # 8) Guardar catálogos y estadísticas

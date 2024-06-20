import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import SnowballStemmer
from sklearn.metrics import precision_score, recall_score, f1_score


def construir_indice_invertido(documentos):
    indice_invertido = {}  # Usa un diccionario estándar
    for doc_id, doc in enumerate(documentos):
        for palabra in doc:
            if palabra not in indice_invertido:
                # Inicializa un conjunto para nuevas palabras
                indice_invertido[palabra] = set()
            # Agrega el doc_id al conjunto
            indice_invertido[palabra].add(doc_id)
    return indice_invertido





def obtener_documentos_relevantes(consulta_procesada, indice_invertido):
    # Inicializar un conjunto con los IDs de los documentos relevantes
    documentos_relevantes = set()
    # Iterar sobre cada palabra de la consulta
    for palabra in consulta_procesada:
        # Buscar la palabra en el índice invertido
        if palabra in indice_invertido:
            # Agregar los IDs de los documentos que contienen la palabra al conjunto de documentos relevantes
            documentos_relevantes.update(indice_invertido[palabra])
    return documentos_relevantes


def procesar_tokens(tokens, stop_words):
    stemmer = SnowballStemmer('english')
    # Eliminar stop words
    tokens_filtrados = [token for token in tokens if token not in stop_words]
    # Aplicar stemming
    tokens_stemmizados = [stemmer.stem(token) for token in tokens_filtrados]
    return tokens_stemmizados


def separar(doc):
    palabras = doc.split()
    return palabras


def limpiar_texto(texto):
    # Eliminar caracteres no deseados (mantener solo letras y espacios)
    texto_limpio = re.sub(r'[^a-zA-Z\s]', '', texto)
    # Normalizar a minúsculas
    texto_limpio = texto_limpio.lower()
    # Eliminar espacios en blanco adicionales
    texto_limpio = re.sub(r'\s+', ' ', texto_limpio).strip()
    return texto_limpio


def metrica_evaluacion(consulta, respuesta, nombres_archivos, documentos):
    # Parsear el archivo cats.txt para obtener la verdad de terreno (ground truth)
    ruta_cats = '../reuters/cats.txt'
    gran_verdad = {}

    with open(ruta_cats, 'r', encoding='latin-1') as f:
        for linea in f:
            if linea.startswith('training/'):
                partes = linea.strip().split()
                # Obtener el ID del documento
                doc_id = int(partes[0].split('/')[1])
                categorias = partes[1:]  # Obtener las categorías
                gran_verdad[doc_id] = categorias

    categorias_consulta = consulta  # Categorías esperadas para la consulta
    documentos_relevantes_esperados = set()

    for doc_id, categorias in gran_verdad.items():
        if any(categoria in categorias_consulta for categoria in categorias):
            documentos_relevantes_esperados.add((doc_id))

    documentos_relevantes_esperados = list(documentos_relevantes_esperados)
    documentos_encontrados_nombres_tfidf = [int(i) for i, _ in respuesta]

    documentos_relevantes_esperados_nuevo = []
    for id_doc in documentos_relevantes_esperados:
        documentos_relevantes_esperados_nuevo.append(
            nombres_archivos.index(id_doc))

    # Evaluar resultados de TF-IDF
    y_true = [1 if i in documentos_relevantes_esperados_nuevo else 0 for i in range(
        len(documentos))]
    y_pred = [1 if i in documentos_encontrados_nombres_tfidf else 0 for i in range(
        len(documentos))]
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1

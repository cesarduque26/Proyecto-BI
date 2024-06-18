import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import SnowballStemmer
import os

def cargar_corpus(directorio):
    # Leer el contenido de los archivos en una lista
    documentos = []
    for archivo in os.listdir(directorio):
        ruta_archivo = os.path.join(directorio, archivo)
        with open(ruta_archivo, 'r', encoding='latin-1') as f:
            documentos.append(f.read())

    return documentos

def cargarstopwords(ruta_stop_words):
    with open(ruta_stop_words, 'r', encoding='latin-1') as f:
        stop_words = set(f.read().split())
    return stop_words

def crearbow(documentos,stop_words):
    vectorizer_bow = CountVectorizer()
    documentos_limpios = [limpiar_texto(doc) for doc in documentos]
    documentos_tokenizados_split = [separar(doc) for doc in documentos_limpios]
    documentos_procesados = [procesar_tokens(doc,stop_words) for doc in documentos_tokenizados_split]
    documentos_procesados_texto = [' '.join(doc) for doc in documentos_procesados]
    bow = vectorizer_bow.fit_transform(documentos_procesados_texto)
    return bow,vectorizer_bow,documentos_procesados

def crearTfidf(documentos,stop_words):
    vectorizer_tfidf = TfidfVectorizer()
    documentos_limpios = [limpiar_texto(doc) for doc in documentos]
    documentos_tokenizados_split = [separar(doc) for doc in documentos_limpios]
    documentos_procesados = [procesar_tokens(doc,stop_words) for doc in documentos_tokenizados_split]
    documentos_procesados_texto = [' '.join(doc) for doc in documentos_procesados]
    Tfidf = vectorizer_tfidf.fit_transform(documentos_procesados_texto)
    return Tfidf,vectorizer_tfidf

def construir_indice_invertido(documentos):
    indice_invertido = {}  # Usa un diccionario estándar
    for doc_id, doc in enumerate(documentos):
        for palabra in doc:
            if palabra not in indice_invertido:
                indice_invertido[palabra] = set()  # Inicializa un conjunto para nuevas palabras
            indice_invertido[palabra].add(doc_id)  # Agrega el doc_id al conjunto
    return indice_invertido

def buscar_bow(consulta,indice_invertido,bow,vectorizer_bow,stop_words):
    # Procesar la consulta
    consulta_procesada = procesar_tokens(separar(limpiar_texto(consulta)),stop_words)
    consulta_vector = vectorizer_bow.transform([' '.join(consulta_procesada)])
    documentos_relevantes = obtener_documentos_relevantes(consulta_procesada,indice_invertido)
    #realizar la matriz de similitud
    bow_2 = bow[list(documentos_relevantes)]
    # Calcular similitud coseno
    similitud_coseno = cosine_similarity(consulta_vector, bow_2).flatten()
    similitud_coseno_id = [(doc_id, similitud_coseno[id]) for id, doc_id in enumerate (documentos_relevantes)]
    similitud_coseno_id.sort(key=lambda x: x[1], reverse=True)
    
    return similitud_coseno_id[:10]

def buscar_Tfidf(consulta,indice_invertido,Tfidf,vectorizer_tfidf,stop_words):
    # Procesar la consulta
    consulta_procesada = procesar_tokens(separar(limpiar_texto(consulta)),stop_words)
    consulta_vector = vectorizer_tfidf.transform([' '.join(consulta_procesada)])
    documentos_relevantes = obtener_documentos_relevantes(consulta_procesada,indice_invertido)
    #realizar la matriz de similitud
    Tfidf2 = Tfidf[list(documentos_relevantes)]
    # Calcular similitud coseno
    similitud_coseno = cosine_similarity(consulta_vector, Tfidf2).flatten()
    similitud_coseno_id = [(doc_id, similitud_coseno[id]) for id, doc_id in enumerate (documentos_relevantes)]
    similitud_coseno_id.sort(key=lambda x: x[1], reverse=True)
    
    return similitud_coseno_id[:10]

def obtener_documentos_relevantes(consulta_procesada,indice_invertido):
    # Inicializar un conjunto con los IDs de los documentos relevantes
    documentos_relevantes = set()
    # Iterar sobre cada palabra de la consulta
    for palabra in consulta_procesada:
        # Buscar la palabra en el índice invertido
        if palabra in indice_invertido:
            # Agregar los IDs de los documentos que contienen la palabra al conjunto de documentos relevantes
            documentos_relevantes.update(indice_invertido[palabra])
    return documentos_relevantes

def procesar_tokens(tokens,stop_words):
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
    return  texto_limpio

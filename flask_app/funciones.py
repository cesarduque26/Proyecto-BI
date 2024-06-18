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
    documentos_limpios = [limpiar_texto(doc) for doc in documentos]
    documentos_tokenizados_split = [separar(doc) for doc in documentos_limpios]
    documentos_procesados = [procesar_tokens(doc,stop_words) for doc in documentos_tokenizados_split]
    documentos_procesados_texto = [' '.join(doc) for doc in documentos_procesados]
    bow = CountVectorizer().fit_transform(documentos_procesados_texto)
    return bow

def crearTfidf(documentos,stop_words):
    documentos_limpios = [limpiar_texto(doc) for doc in documentos]
    documentos_tokenizados_split = [separar(doc) for doc in documentos_limpios]
    documentos_procesados = [procesar_tokens(doc,stop_words) for doc in documentos_tokenizados_split]
    documentos_procesados_texto = [' '.join(doc) for doc in documentos_procesados]
    Tfidf = TfidfVectorizer().fit_transform(documentos_procesados_texto)
    return Tfidf

def construir_indice_invertido(documentos):
    indice_invertido = {}  # Usa un diccionario estándar
    for doc_id, doc in enumerate(documentos):
        for palabra in doc:
            if palabra not in indice_invertido:
                indice_invertido[palabra] = set()  # Inicializa un conjunto para nuevas palabras
            indice_invertido[palabra].add(doc_id)  # Agrega el doc_id al conjunto
    return indice_invertido

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

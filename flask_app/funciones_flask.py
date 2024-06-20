import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import SnowballStemmer
from sklearn.metrics import precision_score, recall_score, f1_score

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
    documentos_relevantes = obtener_documentos_relevantes(consulta_procesada,indice_invertido)
    if documentos_relevantes:
        consulta_vector = vectorizer_tfidf.transform([' '.join(consulta_procesada)])
        #realizar la matriz de similitud
        Tfidf2 = Tfidf[list(documentos_relevantes)]
        # Calcular similitud coseno
        similitud_coseno = cosine_similarity(consulta_vector, Tfidf2).flatten()
        similitud_coseno_id = [(doc_id, similitud_coseno[id]) for id, doc_id in enumerate (documentos_relevantes)]
        similitud_coseno_id.sort(key=lambda x: x[1], reverse=True)
        return similitud_coseno_id[:10]
    else:
        return []

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

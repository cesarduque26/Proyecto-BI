from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import funciones_flask as fun
import pickle

app = Flask(__name__)


# Cargar los datos desde el archivo pickle
with open('../exportado.pkl', 'rb') as f:
    datos = pickle.load(f)

# Extraer los objetos del diccionario
documentos = datos['documentos']
nombres_archivos = datos['nombres_archivos']
stop_words = datos['stop_words']
indice_invertido = datos['indice_invertido']
matriz_bow = datos['matriz_bow']
matriz_tfidf = datos['matriz_tfidf']
vectorizer_bow = datos['vectorizer_bow']
vectorizer_tfidf = datos['vectorizer_tfidf']

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/buscar', methods=['POST'])
def buscar():
    consulta = request.form['consulta']
    radio= request.form['method']
    data={'consulta':consulta,'radio':radio}
    if radio=='bow':
        respuesta=fun.buscar_bow(consulta,indice_invertido,matriz_bow,vectorizer_bow,stop_words)
        resultados = []
        for doc_id,similitud_cos in respuesta:
            lineas = documentos[doc_id].split('\n')  # Dividir el documento en líneas
            titulo = lineas[0]  # La primera línea es el título
            contenido = " ".join(lineas)  # Unir todas las líneas para el contenido completo
            resultados.append({"titulo": titulo, "contenido": contenido,"similitud_coseno":similitud_cos})
        return render_template('resultados.html', resultados=resultados,data=data)
    else:
        respuesta=fun.buscar_Tfidf(consulta,indice_invertido,matriz_tfidf,vectorizer_tfidf,stop_words)
        resultados = []
        for doc_id,similitud_cos in respuesta:
            lineas = documentos[doc_id].split('\n')  # Dividir el documento en líneas
            titulo = lineas[0]  # La primera línea es el título
            contenido = " ".join(lineas)  # Unir todas las líneas para el contenido completo
            resultados.append({"titulo": titulo, "contenido": contenido,"similitud_coseno":similitud_cos})

        return render_template('resultados.html', resultados=resultados,data=data)


if __name__ == '__main__':
    app.run(debug=True, port=5005)
    

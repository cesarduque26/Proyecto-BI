from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import funciones as fun

app = Flask(__name__)

vectorizer_bow = CountVectorizer()
vectorizer_tfidf = TfidfVectorizer()
documentos=fun.cargar_corpus('../reuters/training/')
print(f'Se han cargado {len(documentos)} documentos.')
stop_words=fun.cargarstopwords('../reuters/stopwords')
print(f'Se han cargado {len(stop_words)} stopwords.')
bow,vectorizer_bow=fun.crearbow(documentos,stop_words)

print(f'Se ha creado una matriz BOW de tamaño {bow.shape}.')



@app.route('/')

def index():
    return render_template('index.html')

@app.route('/buscar', methods=['POST'])
def buscar():
    consulta = request.form['consulta']
    radio= request.form['method']
    if radio=='bow':
        respuesta=fun.buscar_bow(consulta,fun.construir_indice_invertido(documentos),bow,vectorizer_bow,stop_words)
        resultados = []
        for doc_id,similitud_cos in respuesta:
            lineas = documentos[doc_id].split('\n')  # Dividir el documento en líneas
            titulo = lineas[0]  # La primera línea es el título
            contenido = " ".join(lineas)  # Unir todas las líneas para el contenido completo
            similitud_coseno=similitud_cos
            resultados.append({"titulo": titulo, "contenido": contenido,"similitud_coseno":similitud_coseno})
        
        return render_template('resultados.html', resultados=resultados)
    




    else:
        print(consulta)
#     resultados = fun.procesar_y_buscar(consulta)
#     return render_template('resultados.html', consulta=consulta, resultados=resultados)

        data={ 'metodo':'Tfidf',
              'consulta':consulta}
        
        return render_template('resultados.html', data=data)


if __name__ == '__main__':
    app.run(debug=True, port=5005)
    

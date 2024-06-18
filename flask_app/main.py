from flask import Flask, render_template, request
import funciones as fun

app = Flask(__name__)


documentos=fun.cargar_corpus('../reuters/training/')
print(f'Se han cargado {len(documentos)} documentos.')
stop_words=fun.cargarstopwords('../reuters/stopwords')
print(f'Se han cargado {len(stop_words)} stopwords.')
bow=fun.crearbow(documentos,stop_words)
print(f'Se ha creado una matriz BOW de tamaño {bow.shape}.')


@app.route('/')

def index():
    return render_template('index.html')

@app.route('/buscar', methods=['POST'])
def buscar():
    consulta = request.form['consulta']
    radio= request.form['method']
    if radio=='bow':
        data={ 'metodo':'bow',
              'consulta':consulta}
        
        return render_template('resultados.html', data=data)
    




    else:
        print(consulta)
#     resultados = fun.procesar_y_buscar(consulta)
#     return render_template('resultados.html', consulta=consulta, resultados=resultados)

        data={ 'metodo':'Tfidf',
              'consulta':consulta}
        
        return render_template('resultados.html', data=data)


if __name__ == '__main__':
    app.run(debug=True, port=5005)
    

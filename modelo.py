from flask import Flask, render_template

app = Flask(__name__) 

data = {
         'titulo01': 'fundamentdos e flask',
         'descripcion01': 'cesar',
         'titulo02': 'fundamentdos e flask',
         'descripcion02': 'cesar',
         'titulo01': 'fundamentdos e flask',
         'descripcion01': 'cesar',
         'titulo01': 'fundamentdos e flask',
         'descripcion01': 'cesar',
         'edad': 23
    }
def index():
    # return render_template('index.html',titulo='cesar duque') #render_template es para renderizar un archivo html y tambien se le puede pasar variables
    
    # render_template es para renderizar un archivo html y tambien se le puede pasar variables
    return render_template('index.html', datos=data)

def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.add_url_rule('/', view_func=index)
    app.add_url_rule('/home', view_func=home)
    # app.add_url_rule('/', view_func=hello_world)

    # debug=True para que se actualice automaticamente el servidor cada vez que hagamos cambios en el codigo , port=5005 para cambiar el puerto por defecto 5000 esto es opcional
    app.run(debug=True, port=5005)

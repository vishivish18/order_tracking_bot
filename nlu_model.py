from __future__ import unicode_literals

from flask import Flask, abort, request 
from flask import jsonify

from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer
from rasa_nlu.model import Metadata, Interpreter



class PrefixMiddleware(object):
    def __init__(self, app, prefix=''):
      self.app = app
      self.prefix = prefix
    
    def __call__(self, environ, start_response):
      if environ['PATH_INFO'].startswith(self.prefix):
        environ['PATH_INFO'] = environ['PATH_INFO'][len(self.prefix):]
        environ['SCRIPT_NAME'] = self.prefix
        return self.app(environ, start_response)
      else:
        start_response('404', [('Content-Type', 'text/plain')])
        return ["This url does not belong to the app.".encode()]


# app = web.application(urls, globals())
app = Flask(__name__)
app.wsgi_app = PrefixMiddleware(app.wsgi_app, prefix='/api/nlp/v0.1')

# app.config['APPLICATION_ROOT'] = '/api/ms/v3.0'
# wsgiapp = app.wsgifunc()

@app.route("/ignition")
def ignition():
    return jsonify({"Engine Status" : "Chatbot is live"}), 200


@app.route("/train")
def train_nlu(data ='./data/data.json', config = './config_sapcy.json', model_dir = './models/nlu'):
	training_data = load_data(data)
	trainer = Trainer(RasaNLUConfig(config))
	trainer.train(training_data)
	model_directory = trainer.persist(model_dir, fixed_model_name = 'stagebot')
	return jsonify({"Training" : model_directory }), 200

@app.route("/query")
def run_nlu():
	query = request.args['q']
	print(query)
	interpreter = Interpreter.load('./models/nlu/default/stagebot', RasaNLUConfig('config_sapcy.json'))
	response = interpreter.parse(query)
	return jsonify({"Engine Status" : response}), 200






# if __name__  == "__main__":
# 	#train_nlu('./data/data.json','./config_sapcy.json','./models/nlu')
# 	run_nlu();


if __name__ == "__main__":
	app.run(host='0.0.0.0', port=8080)




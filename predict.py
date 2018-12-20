# predict patch image's count using trained fully connected model

from keras.models import model_from_json
import scipy.io as sci
from features2X import features2X

test_data_dir = 'processed/raw_input.mat'
print('loading test data...')
test_data = sci.loadmat(test_data_dir)
X_test = features2X(test_data['features'][0])

# load trained model from disk
json_file = open('../model/model_B_SHT.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("../model/model_B_SHT.h5")
print("Loaded model from disk")

print(model.summary())
# exit() 

model.compile(optimizer='Adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

predictions = model.predict(X_test, batch_size=1000, verbose=0)
sci.savemat('processed/predictions.mat', {'predictions': predictions})

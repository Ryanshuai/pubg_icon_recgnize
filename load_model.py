from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

prediction = CustomImagePrediction()
prediction.setModelTypeAsSqueezeNet()
prediction.setModelPath(os.path.join(execution_path, "model_ex-015_acc-0.998549.h5"))
prediction.setJsonPath(os.path.join(execution_path, "model_class.json"))
prediction.loadModel(num_objects=21)

test_dir = "my_test"

for image_name in os.listdir(test_dir):
    image_path = os.path.join(test_dir, image_name)

    predictions, probabilities = prediction.predictImage(image_path, result_count=5)
    print(image_name + "----->" + str(predictions) + " : " + str(probabilities))

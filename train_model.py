from imageai.Prediction.Custom import ModelTraining

model_trainer = ModelTraining()
model_trainer.setModelTypeAsSqueezeNet()
# model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory("icon_dataset")
# model_trainer.trainModel(num_objects=21, num_experiments=100, enhance_data=True, initial_learning_rate=1e-4,
#                          batch_size=128, training_image_size=100,
#                          show_network_summary=True, save_full_model=True,
#                          continue_from_model="model_ex-048_acc-0.999540.h5",)


model_trainer.trainModel(num_objects=21, num_experiments=100, enhance_data=False, initial_learning_rate=1e-4,
                         batch_size=32, training_image_size=100,
                         show_network_summary=True, save_full_model=True,
                         )

from keras.models import Model


def create_detector_model_from_trained(model):
    return Model(inputs=model.input[0], outputs=model.get_layer('DetectionLayer').output)


def create_training_model_from_detector(model):
    return Model(inputs=model.input, outputs=model.get_layer('leaky_re_lu_22').output)
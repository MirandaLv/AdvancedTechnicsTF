
import tensorflow as tf


def feature_extraction(inputs):

    feature_extractor_layer = tf.keras.applications.resnet.ResNet50(
        input_shape=(224,224,3),
        include_top=False, # top of the model is the end that predicts the output of the model, the top of
        # last layer has a 1000 neuron dense classification layer, we dont want to sue this layer because it
        # predicts 1000 classes
        weights='imagenet')(input) # use the imagenet learned weights, starting point weights are from imagenet
    return feature_extractor_layer

def classifier(inputs):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(10, activation='softmax', name='classification')(x)
    return x

def final_model(inputs):

    # the CIFA dataset is 32*32, but the ResNet requires 224, so unsample the image to 224
    resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)
    resnet_feature_extractor = feature_extraction(resize)
    classification_output = classifier(resnet_feature_extractor)

    return classification_output


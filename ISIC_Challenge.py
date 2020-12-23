import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import math as math
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# download CUDA 10.1
#physical_device = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_device[0], True)

train_examples = 20225
test_examples = 2551
validation_examples = 2555
img_height = img_width = 224
batch_size = 32

model = keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/4", trainable=True),
    layers.Dense(1, activation="sigmoid"),
])

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    zoom_range=(0.95, 0.95),
    horizontal_flip=True,
    vertical_flip=True,
    data_format="channels_last",
    dtype=tf.float32,  # no validation split here cuz all of this data augmentation will be applied to the
    # validation set....big no no
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255, dtype=tf.float32)
test_datagen = ImageDataGenerator(rescale=1.0 / 255, dtype=tf.float32)

# training
train_gen = train_datagen.flow_from_directory(
    "D:/data/train",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)

# validation
validation_gen = validation_datagen.flow_from_directory(
    "D:/data/validation",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)

# test
test_gen = test_datagen.flow_from_directory(
    "D:/data/test",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)

# accuracy here will be a flawed metric cuz the classes are imbalanced and the data is imbalanced
# in otherwords, 10% of the photos could be cancer while the other 90% could be skin lesions
# With that said, we'll have to use a different metric using precision, recall and auc and accuracy
METRICS = [
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.Precision(name="precision"), # computes precision of predictions w/ regards to the label
    keras.metrics.Recall(name="recall"),# computes recall: examines true positive, false negative, false_positives, and true_negatives and calls combined with sample weights
    keras.metrics.AUC(name="auc"), # Area Under Curve via Riemann sum: The area under the ROC-curve is therefore computed using the height of the recall values by the false positive rate, while the area under the PR-curve is the computed using the height of the precision values by the recall.
]

model.compile(
    optimizer=keras.optimizers.Adam(lr=3e-4),
    loss=[keras.losses.BinaryCrossentropy(from_logits=False)],
    metrics=METRICS
)

model.fit(
    train_gen,
    epochs=15,
    steps_per_epoch=train_examples // batch_size,
    validation_data=validation_gen,
    validation_steps=validation_examples,
    #callbacks=[keras.callbacks.ModelCheckpoint("cancer_detection_model")] will save a model every epoch
)

# calculate true positive and false positive rate
def plot_roc(labels, data):
    predictions = model.predict(data)
    fp, tp, _ = roc_curve(labels, predictions)

    plt.plot(100*fp, 100*tp)
    plt.xlabel("False Positive in %")
    plt.ylabel("True Positives in %")
    plt.show()

test_labels = np.array([])
num_batches = 0

for _, y in test_gen:
    test_labels = np.append(test_labels, y)
    num_batches += 1
    if num_batches == math.ceil(test_examples / batch_size):
        break

# give us ROC curve on the graph
plot_roc(test_labels, test_gen)

model.evaluate(validation_gen, verbose=2)
model.evaluate(test_gen, verbose=2)

model.save_weights('saved_model/')

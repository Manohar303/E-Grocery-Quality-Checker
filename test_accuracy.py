
# Imports
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from PIL import Image, ImageOps

# Input Directories
model_path = "/Users/ashok_mb/Downloads/tomato/model.savedmodel"
test_data_dir = "/Users/ashok_mb/Downloads/test_data/tomato"

# Model Input Configs
batch_size = 32
img_height = 224
img_width = 224
class_map = {
    0: "average",
    1: "bad",
    2: "good",
}

# Load test data
test_data = image_dataset_from_directory(
    test_data_dir,
    color_mode='rgb',
    shuffle=False,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)
# Load the trained Model
model = load_model(model_path)
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# Convert images to desired input format


def pre_process_image(img):
    size = (img_height, img_width)
    data = np.ndarray(shape=(1, img_height, img_width, 3), dtype=np.float32)
    image = Image.open(img).convert('RGB')
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    return data

# Predict the model


def predict(_data):
    images = _data.file_paths
    actuals, predictions = [], []
    for img in images:
        actuals.append(img.split("/")[-2])
        data = pre_process_image(img)
        pred_y = model.predict(data)
        pred_y = pred_y.tolist()[0]
        prediction = pred_y.index(max(pred_y))
        prediction = class_map[prediction]
        predictions.append(prediction)
    return actuals, predictions


actuals, predictions = predict(test_data)
print(predictions[-10:])
print(actuals[-10:])
# Accuracy and Confusion Matrix
print(confusion_matrix(actuals, predictions))
# Model Accuracy
accuracy = accuracy_score(actuals, predictions)
print("Model accuracy for test dataset is {:.2f}%".format(accuracy*100))

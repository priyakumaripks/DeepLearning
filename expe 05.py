# XAI demonstration for an image classification model.
# Install: pip install tensorflow matplotlib lime shap
# Set MODEL_PATH and IMAGE_PATH before running.
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

MODEL_PATH = "model.keras"
IMAGE_PATH = "sample.jpg"
SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)
img = Image.open(IMAGE_PATH).convert("RGB").resize(SIZE)
x = np.array(img).astype("float32")
batch = np.expand_dims(x, axis=0)
prediction = model.predict(batch, verbose=0)
class_id = int(np.argmax(prediction[0]))
print("Predicted class index:", class_id)

# A) Occlusion: hide image patches and measure score change
def occlusion(image, patch=32):
    original = model.predict(image[None, ...], verbose=0)[0, class_id]
    heat = np.zeros(image.shape[:2])
    for y in range(0, image.shape[0], patch):
        for x0 in range(0, image.shape[1], patch):
            temp = image.copy()
            temp[y:y+patch, x0:x0+patch] = 0
            score = model.predict(temp[None, ...], verbose=0)[0, class_id]
            heat[y:y+patch, x0:x0+patch] = original - score
    return heat

# B) Integrated Gradients
def integrated_gradients(model, image, steps=50):
    inp = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
    baseline = tf.zeros_like(inp)
    alphas = tf.linspace(0.0, 1.0, steps + 1)
    total = tf.zeros_like(inp)
    for a in alphas:
        scaled = baseline + a * (inp - baseline)
        with tf.GradientTape() as tape:
            tape.watch(scaled)
            score = model(scaled, training=False)[:, class_id]
        total += tape.gradient(score, scaled)
    avg_grad = total / (steps + 1)
    return ((inp - baseline) * avg_grad)[0].numpy().sum(axis=-1)

# C) Direct propagation: show the model's forward-pass class scores.
# This is a prediction/activation inspection, not a standard named XAI method.
print("Direct forward-pass scores:", prediction[0])

# D) Grad-CAM (requires a convolutional layer name)
def grad_cam(model, image, conv_layer_name):
    grad_model = tf.keras.Model(
        model.inputs,
        [model.get_layer(conv_layer_name).output, model.output]
    )
    inp = tf.convert_to_tensor(image[None, ...])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(inp, training=False)
        score = preds[:, class_id]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_out[0] * weights[0], axis=-1)
    cam = tf.maximum(cam, 0)
    return (cam / (tf.reduce_max(cam) + 1e-8)).numpy()

# E) Grad-CAM++ approximation using positive gradient powers
def grad_cam_plus_plus(model, image, conv_layer_name):
    grad_model = tf.keras.Model(
        model.inputs,
        [model.get_layer(conv_layer_name).output, model.output]
    )
    inp = tf.convert_to_tensor(image[None, ...])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(inp, training=False)
        score = preds[:, class_id]
    grads = tape.gradient(score, conv_out)
    grads2, grads3 = grads ** 2, grads ** 3
    sum_acts = tf.reduce_sum(conv_out, axis=(1, 2), keepdims=True)
    alpha = grads2 / (2 * grads2 + sum_acts * grads3 + 1e-8)
    weights = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=(1, 2))
    cam = tf.reduce_sum(conv_out[0] * weights[0], axis=-1)
    cam = tf.maximum(cam, 0)
    return (cam / (tf.reduce_max(cam) + 1e-8)).numpy()

# Display image and model-agnostic occlusion/gradient maps
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.imshow(img); plt.title("Original"); plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(occlusion(x), cmap="jet"); plt.title("Occlusion"); plt.axis("off")
plt.subplot(1, 3, 3); plt.imshow(integrated_gradients(model, x), cmap="jet"); plt.title("Integrated Gradients"); plt.axis("off")
plt.tight_layout(); plt.show()

# LIME: local explanation
try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        x.astype("double"),
        lambda images: model.predict(np.array(images), verbose=0),
        top_labels=1, hide_color=0, num_samples=500
    )
    temp, mask = explanation.get_image_and_mask(
        class_id, positive_only=True, num_features=5, hide_rest=False
    )
    plt.imshow(mark_boundaries(temp / 255.0, mask))
    plt.title("LIME"); plt.axis("off"); plt.show()
except ImportError:
    print("For LIME: pip install lime scikit-image")

# SHAP: model-agnostic image explanation (can be computationally expensive)
try:
    import shap
    masker = shap.maskers.Image("blur(32,32)", x.shape)
    explainer = shap.Explainer(
        lambda images: model.predict(np.array(images), verbose=0),
        masker, output_names=[str(i) for i in range(prediction.shape[1])]
    )
    shap_values = explainer(batch, max_evals=100, batch_size=10)
    shap.image_plot(shap_values)
except ImportError:
    print("For SHAP: pip install shap")

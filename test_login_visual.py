import pytest
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import imagehash
from PIL import Image


@pytest.fixture(scope="session")
def vgg_model():
    return VGG16(weights='imagenet', include_top=False, pooling='avg')


def get_features(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return model.predict(preprocess_input(x))


def test_semantic_login_ui(page, vgg_model):
    # 1. Capture Current UI
    page.goto("http://localhost:5000/")
    current_path = "templates/current_login.png"
    baseline_path = "templates/baseline_login.png"
    page.screenshot(path=current_path)

    # 2. Extract and Compare Features (VGG16)
    feat_baseline = get_features(vgg_model, baseline_path)
    feat_current = get_features(vgg_model, current_path)

    similarity = cosine_similarity(feat_baseline, feat_current)[0][0]
    print("Visual Similarity Score: ", similarity)

    # 3. Perceptual Hashing (Hamming Distance)
    hash_baseline = imagehash.phash(Image.open(baseline_path))
    hash_current = imagehash.phash(Image.open(current_path))
    hamming_dist = hash_baseline - hash_current
    print("Hamming Distance: ", hamming_dist)

    # 4. Assert Similarity
    # Pass if semantic similarity is high OR if structural similarity is high (low hamming distance)
    assert similarity > 0.90 or hamming_dist < 5, \
        f"Visual regression detected! Score: {similarity}, Hamming Dist: {hamming_dist}"

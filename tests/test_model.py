import os

from src.model import train_model


def test_model_accuracy():
    minimum = float(os.environ.get('MIN_MODEL_ACCURACY', '0.80'))
    accuracy = train_model()
    assert accuracy >= minimum, f'Model accuracy {accuracy:.4f} is below the required {minimum:.2f}.'

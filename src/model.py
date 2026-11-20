from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_model():
    """Fit a scaled Iris classifier and return its fixed holdout accuracy."""
    features, labels = load_iris(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels)
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
    pipeline.fit(train_x, train_y)
    return float(accuracy_score(test_y, pipeline.predict(test_x)))


if __name__ == '__main__':
    print(f'Model Accuracy: {train_model():.4f}')

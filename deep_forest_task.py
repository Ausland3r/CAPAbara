# deep_forest_task.py
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from deepforest import CascadeForestClassifier


def main(input_csv, output_csv):
    # Загрузка данных
    data = pd.read_csv(input_csv, encoding='utf-8')

    # Использование KMeans для определения пороговых значений
    features = ['Additions', 'Deletions', 'Total Changes', 'File Changes', 'Time Since Last Commit']
    kmeans = KMeans(n_clusters=2, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data[features])

    # Обработка данных для модели глубокого леса
    X = data[features]
    y = data['Cluster']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели глубокого леса
    gc_model = CascadeForestClassifier(random_state=42)
    gc_model.fit(X_train, y_train)

    # Оценка модели глубокого леса
    y_pred = gc_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Сохранение результатов в CSV
    data['GCForest_Prediction'] = gc_model.predict(data[features])
    data.to_csv(output_csv, index=False)

    print(f"GCForest Accuracy: {accuracy}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python deep_forest_task.py <input_csv> <output_csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

import csv
import subprocess
import sys
import os
from github import Github
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
from sklearn.cluster import KMeans

# Чтение конфигурации из файла
config = {}
with open('config.txt', 'r') as file:
    for line in file:
        name, value = line.strip().split('=')
        config[name] = value

access_token = config['access_token']
repos = config['repos'].split(',')

# Аутентификация с использованием токена доступа
g = Github(access_token)

# Создание файла для записи данных
with open('repository_data.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Repo', 'Commit SHA', 'Author', 'Date', 'Message', 'Additions', 'Deletions', 'Total Changes', 'File Changes', 'Time Since Last Commit', 'Has CAPA'])

    for repo_name in repos:
        repo = g.get_repo(repo_name)
        commits = repo.get_commits()

        previous_commit_date = None
        for commit in commits:
            sha = commit.sha
            author = commit.commit.author.name
            date = commit.commit.author.date
            message = commit.commit.message
            stats = commit.stats
            additions = stats.additions
            deletions = stats.deletions
            total_changes = stats.total
            files_changed = commit.files

            if previous_commit_date:
                time_since_last_commit = abs((date - previous_commit_date).days)
            else:
                time_since_last_commit = 0

            previous_commit_date = date
            has_capa = 0  # Изначально все коммиты не CAPA
            writer.writerow([repo_name, sha, author, date, message, additions, deletions, total_changes, len(files_changed), time_since_last_commit, has_capa])

print("Данные успешно сохранены в repository_data.csv")

# Загрузка данных из файла
data = pd.read_csv('repository_data.csv', encoding='utf-8')

# Использование KMeans для определения пороговых значений
features = ['Additions', 'Deletions', 'Total Changes', 'File Changes', 'Time Since Last Commit']
kmeans = KMeans(n_clusters=2, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[features])

# Вычисление пороговых значений на основе кластеров
cluster_centers = kmeans.cluster_centers_
ADDITION_THRESHOLD = cluster_centers[:, 0].mean()
DELETION_THRESHOLD = cluster_centers[:, 1].mean()
TOTAL_CHANGE_THRESHOLD = cluster_centers[:, 2].mean()
FILE_CHANGE_THRESHOLD = cluster_centers[:, 3].mean()
TIME_SINCE_LAST_COMMIT_THRESHOLD = cluster_centers[:, 4].mean()

# Обработка данных для основной модели
X = data[features]
y = data['Cluster']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение основной модели случайного леса
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Оценка основной модели случайного леса
rf_y_pred = rf_model.predict(X_test)
print("RandomForest Accuracy:", accuracy_score(y_test, rf_y_pred))

# Сохранение важностей признаков в CSV
importance_df = pd.DataFrame({'Feature': features, 'Importance': rf_model.feature_importances_})
importance_df.to_csv('feature_importances.csv', index=False)

input_csv = 'repository_data.csv'
output_csv = 'repository_data_with_predictions.csv'
python_venv = os.path.join(os.getcwd(), 'myenv', 'Scripts', 'python.exe')
if not os.path.exists(python_venv):
    print(f"Virtual environment not found at {python_venv}")
    sys.exit(1)

result = subprocess.run([python_venv, 'deep_forest_task.py', input_csv, output_csv], capture_output=True, text=True)
print(result.stdout)
print(result.stderr, file=sys.stderr)

if result.returncode == 0:
    data_with_predictions = pd.read_csv(output_csv)
    print("Результаты предсказаний загружены успешно.")
else:
    print("Ошибка при выполнении подзадачи.")

# Функция генерации рекомендаций CAPA
def generate_capa_recommendations(data, ADDITION_THRESHOLD, DELETION_THRESHOLD, TOTAL_CHANGE_THRESHOLD, FILE_CHANGE_THRESHOLD, TIME_SINCE_LAST_COMMIT_THRESHOLD):
    recommendations = []
    for index, commit_data in data.iterrows():
        suggestion = []
        if commit_data['Additions'] > ADDITION_THRESHOLD:
            suggestion.append(' | Review large additions |')
        if commit_data['Deletions'] > DELETION_THRESHOLD:
            suggestion.append(' | Review large deletions (Many deletions may indicate a fix) |')
        if commit_data['Total Changes'] > TOTAL_CHANGE_THRESHOLD:
            suggestion.append(' | Review total changes for potential issues (Too many changes since last commit) |')
        if commit_data['File Changes'] > FILE_CHANGE_THRESHOLD:
            suggestion.append(' | Review changes in multiple files (Many files updated, expected compatibility issues) |')
        if abs(commit_data['Time Since Last Commit']) > abs(TIME_SINCE_LAST_COMMIT_THRESHOLD):
            suggestion.append('| Review large time gap between commits (Large time gap between commits may indicate a slow development process) |')
        if len(commit_data['Message']) < 15 and commit_data['Total Changes'] > TOTAL_CHANGE_THRESHOLD:
            suggestion.append('| Commit message is too short, consider providing more details |')
        if not suggestion:
            suggestion.append('No specific recommendations')

        recommendations.append({
            'Repo': commit_data['Repo'],
            'Commit SHA': commit_data['Commit SHA'],
            'Message': commit_data['Message'],
            'Suggestion': '; '.join(suggestion)
        })

    return recommendations

# Выявление аномалий и генерация рекомендаций
recommendations = generate_capa_recommendations(data_with_predictions, ADDITION_THRESHOLD, DELETION_THRESHOLD, TOTAL_CHANGE_THRESHOLD, FILE_CHANGE_THRESHOLD, TIME_SINCE_LAST_COMMIT_THRESHOLD)
recommendations_df = pd.DataFrame(recommendations)

# Сохранение рекомендаций в CSV
recommendations_df.to_csv('recommendations.csv', index=False)
data['Has CAPA'] = data['Commit SHA'].isin(recommendations_df[recommendations_df['Suggestion'] != 'No specific recommendations']['Commit SHA']).astype(int)

# Анализ данных по авторам
author_analysis = data.groupby(['Repo', 'Author']).agg({
    'Commit SHA': 'count',
    'Additions': 'sum',
    'Deletions': 'sum',
    'Total Changes': 'sum',
    'File Changes': 'sum',
    'Time Since Last Commit': 'mean',
    'Has CAPA': 'sum'
}).reset_index()
print(author_analysis)
author_analysis.rename(columns={'Commit SHA': 'Commit Count'}, inplace=True)
author_analysis['CAPA Rate'] = author_analysis['Has CAPA'] / author_analysis['Commit Count']

author_analysis.to_csv('author_analysis.csv', index=False)



# Удаление существующей ветки, если она есть
repo = g.get_repo(repos[0])
try:
    ref = repo.get_git_ref('heads/capa-recommendations')
    ref.delete()
except:
    pass

# Создание файла с рекомендациями в новой ветке
repo = g.get_repo(repos[0])
source = repo.get_branch("main")
repo.create_git_ref(ref='refs/heads/capa-recommendations', sha=source.commit.sha)
repo.create_file(path="capa_suggestions.csv", message="Add CAPA suggestions", content=recommendations_df.to_csv(index=False), branch="capa-recommendations")
pr = repo.create_pull(title="Add CAPA suggestions", body="This PR adds CAPA suggestions based on the recent analysis.", head="capa-recommendations", base="main")

print("Pull request успешно создан:", pr.html_url)

# Запуск dashboard.py
print("Запуск дашборда...")
subprocess.run([sys.executable, 'dashboard.py'])
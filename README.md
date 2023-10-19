# svmt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# Загрузка датасета о пассажирах Титаника
data = pd.read_csv('titanic.csv')

# Заменяем пропущенные значения в возрасте средним значением
data['Age'].fillna(data['Age'].mean(), inplace=True)

# Преобразование категориальных признаков (Sex) в числовые с помощью кодирования One-Hot
encoder = OneHotEncoder(sparse_output=False)
sex_encoded = encoder.fit_transform(data[["Sex"]])
data_encoded = pd.concat([data, pd.DataFrame(sex_encoded, columns=encoder.get_feature_names_out(["Sex"]))], axis=1)
data_encoded = data_encoded.drop(columns=["Sex"])
# Разделение данных на обучающую и тестовую выборки
X = data_encoded[['Pclass', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare', 'Sex_female', 'Sex_male']]
y = data_encoded['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели Decision Tree
decision_tree_classifier = DecisionTreeClassifier()

# Обучение модели на данных
decision_tree_classifier.fit(X_train, y_train)

# Предсказание классов на тестовых данных
y_pred = decision_tree_classifier.predict(X_test)

# Оценка производительности модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Decision Tree for Titanic Survival Classification: {accuracy}")

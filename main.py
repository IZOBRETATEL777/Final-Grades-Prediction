import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

_ml_model = LinearRegression()
_features = ["Intellectual Abilities", "Hours", "Special Abilities", "Material well-being", "Final Grade"]


def split_and_test(X, y, test_size=0.2):
    global _ml_model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    _ml_model.fit(X_train, y_train)
    print("Training score:", _ml_model.score(X_train, y_train))


def predict(X, y):
    global _ml_model
    predict_X = _get_user_input()
    _ml_model.fit(X, y)
    predictions = _ml_model.predict(predict_X)
    print("Predictions:", predictions)


def _get_user_input():
    print("Evaluate your intellectual abilities (0-2)")
    intellectual = int(input("Enter your choice: "))
    print(
        "How many hours per week did you allocate for "
          "the course/subject to study it "
          "(apart from hours of university lectures or labs)? (1-8)"
    )
    hours = int(input("Enter your choice: "))
    print("Do you have special abilities/aptitude for this subject? (0-1)")
    special = int(input("Enter your choice: "))
    print("How do you evaluate your material well-being? (0-2)")
    material = int(input("Enter your choice: "))
    print("What was your final grade for that subject? Attention, German grading system is used! (1-5)")
    grade = int(input("Enter your choice: "))

    return [intellectual, hours, special, material, grade]


def print_model_info(X, y):
    global _ml_model
    print("Coefficients:", _ml_model.coef_)
    print("Intercept:", _ml_model.intercept_)
    plt.ylabel("Grade")
    for feature in _features[:-1]:
        plt.title("Grade vs " + feature)
        plt.xlabel(feature)
        plt.scatter(X[feature], y)
        plt.plot(X[feature], _ml_model.predict(X), color='red')
        plt.show()


def main():
    pd_data = pd.read_csv('data/data.csv')
    pd_data = pd_data.drop(pd_data.columns[0], axis=1)
    pd_data = pd_data.replace(to_replace=['Yes', 'No'], value=[1, 0])
    pd_data.columns = _features
    pd_data['Intellectual Abilities'] = pd_data['Intellectual Abilities'].astype(int)
    pd_data['Hours'] = pd_data['Hours'].astype(int)
    pd_data['Material well-being'] = pd_data['Material well-being'].astype(int)


    X = pd_data[_features[:-1]]
    y = pd_data[_features[-1]]
    split_and_test(X, y)
    # print_model_info(X, y)
    # predict(X, y)

if __name__ == '__main__':
    main()
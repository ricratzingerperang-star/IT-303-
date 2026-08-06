#Student
from pathlib import Path, os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

file_path = Path(__file__).with_name("student_performance.csv")

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def prompt(prompt, value_type, minimum=None):
    while True:
        try:
            value = value_type(input(prompt))
            if minimum is not None and value < minimum:
                print(f"Please enter a value of at least {minimum}.")
                continue
            return value
        except ValueError:
            print("Please enter a valid number.")

                                                       
def problem1():
    df = pd.read_csv(file_path)

    X = df[[
        "study_hours_per_week",
        "attendance_percent",
        "previous_score",
        "practice_tests"
    ]]
    y = df["final_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n===== Problem 1 =====")
    print("MAE:", mean_absolute_error(y_test, predictions))
    print("R²:", r2_score(y_test, predictions))
    
    study_hours = prompt("study_hours: ", float, minimum=1)
    attendance  = prompt("attendance: ", float, minimum=1)
    score = prompt("score: ", float, minimum=1)
    tests = prompt("tests: ", float, minimum=1)
    
    student = pd.DataFrame(
        [[study_hours, attendance, score, tests]],
        columns=[
            "study_hours_per_week",
            "attendance_percent",
            "previous_score",
            "practice_tests"
        ]
    )

    predicted_score = model.predict(student)[0]
    print("Predicted Final Score:", round(predicted_score, 2))


def problem2():
    df = pd.read_csv(file_path)

    y = df["final_score"]

    
    X_a = df[["study_hours_per_week",
            "attendance_percent",
            "previous_score",
            "practice_tests"]]

    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        X_a, y, test_size=0.2, random_state=42
    )

    model_a = LinearRegression()
    model_a.fit(X_train_a, y_train_a)

    
    X_b = df[[
        "study_hours_per_week",
        "attendance_percent",
        "previous_score",
        "practice_tests"
    ]]

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_b, y, test_size=0.2, random_state=42
    )

    model_b = LinearRegression()
    model_b.fit(X_train_b, y_train_b)
    
    print("=== Model A ===")
    study_hours = prompt("study_hours: ", float, minimum=1)
    attendance  = prompt("attendance: ", float, minimum=1)
    score = prompt("score: ", float, minimum=1)
    tests = prompt("tests: ", float, minimum=1)

    student_a = pd.DataFrame(
        [[study_hours, attendance, score, tests]],
        columns=[
            "study_hours_per_week",
            "attendance_percent",
            "previous_score",
            "practice_tests"
        ]
    )
    
    print("=== Model B ===")
    study_hours = prompt("study_hours: ", float, minimum=1)
    attendance  = prompt("attendance: ", float, minimum=1)
    score = prompt("score: ", float, minimum=1)
    tests = prompt("tests: ", float, minimum=1)

    student_b = pd.DataFrame(
        [[study_hours, attendance, score, tests]],
        columns=[
            "study_hours_per_week",
            "attendance_percent",
            "previous_score",
            "practice_tests"
        ]
    )

    prediction_a = model_a.predict(student_a)[0]
    prediction_b = model_b.predict(student_b)[0]

    print("\n===== Problem 2 =====")
    print("Model A Prediction:", round(prediction_a, 2))
    print("Model B Prediction:", round(prediction_b, 2))

    print("\nObservation:")
    print("The predictions from Model A and Model B are different.")

    print("\nExplanation:")
    print("Model A uses only study hours.")
    print("Model B uses study hours, attendance, previous score, and practice tests.")
    print("Using more features usually produces a more accurate prediction.")


def menu():
    while True:
        print(" Student Performance Menu")
        print("1. Problem 1 - Predict Student Final Scores")
        print("2. Problem 2 - Compare Student Score Models")
        print("3. Exit")
        choice = input("Enter your choice (1-2): ")

        if choice == "1":
            problem1() 
            input("Press to continue...")
            clear()
        elif choice == "2":
            problem2()
            input("Press to continue...")
            clear()
        elif choice == "3":
            print("Exiting the program...")
            break    
        else:
            print("Invalid menu choice.")


if __name__ == "__main__":
    menu()
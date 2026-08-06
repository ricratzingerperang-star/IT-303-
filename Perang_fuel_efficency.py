#Vehicle
from pathlib import Path, os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

file_path = Path(__file__).with_name("vehicle_fuel_efficiency.csv")

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

def problem3():
    df = pd.read_csv(file_path)

    X = df[[
        "engine_size_l",
        "horsepower",
        "vehicle_age_years",
        "mileage_km"
    ]]
    y = df["fuel_efficiency_kmpl"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y, 
        test_size=0.2, 
        random_state=42,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n===== Problem 3 =====")
    print("MAE:", mean_absolute_error(y_test, predictions))
    print("R²:", r2_score(y_test, predictions))

    engine = prompt("Engine_size: ", float, minimum=1)
    horsepower  = prompt("horsepower: ", float, minimum=1)
    age = prompt("vehicle_age_year: ", float, minimum=1)
    mileage = prompt("miliage_km: ", float, minimum=1)
    
    vehicle = pd.DataFrame(
        [[engine, horsepower, age, mileage]],
        columns=[
            "engine_size_l",
            "horsepower",
            "vehicle_age_years",
            "mileage_km"
        ]
    )

    predicted = model.predict(vehicle)[0]
    print("Predicted Fuel Efficiency:", round(predicted, 2), "km/L")


def problem4():
    df = pd.read_csv(file_path)

    y = df["fuel_efficiency_kmpl"]

    
    X_a = df[[
        "engine_size_l",
        "horsepower",
        "vehicle_age_years"
    ]]

    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        X_a, y, test_size=0.2, random_state=42
    )

    model_a = LinearRegression()
    model_a.fit(X_train_a, y_train_a)

    pred_a = model_a.predict(X_test_a)

    mae_a = mean_absolute_error(y_test_a, pred_a)
    r2_a = r2_score(y_test_a, pred_a)

    
    X_b = df[[
        "engine_size_l",
        "horsepower",
        "vehicle_age_years",
        "mileage_km"
    ]]

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_b, y, test_size=0.2, random_state=42
    )

    model_b = LinearRegression()
    model_b.fit(X_train_b, y_train_b)

    pred_b = model_b.predict(X_test_b)

    mae_b = mean_absolute_error(y_test_b, pred_b)
    r2_b = r2_score(y_test_b, pred_b)

    print("\n===== Problem 4 =====")
    print("\nModel A")
    print("MAE:", mae_a)
    print("R²:", r2_a)

    print("\nModel B")
    print("MAE:", mae_b)
    print("R²:", r2_b)

    if mae_b < mae_a:
        print("\nObservation:")
        print("Mileage improves the prediction.")

        print("\nExplanation:")
        print("Adding mileage gives the model more information about the vehicle's condition.")

        engine = prompt("Engine_size: ", float, minimum=1)
        horsepower  = prompt("Horsepower: ", float, minimum=1)
        age = prompt("vehicle_age_years: ", float, minimum=1)
        mileage = prompt("Mileage: ", float, minimum=1)
        vehicle = pd.DataFrame(
            [[engine, horsepower, age, mileage]],
            columns=[
                "engine_size_l",
                "horsepower",
                "vehicle_age_years",
                "mileage_km"
            ]
        )

        prediction = model_b.predict(vehicle)[0]

    else:
        
        engine = prompt("Engine_size: ", float, minimum=1)
        horsepower  = prompt("Horsepower: ", float, minimum=1)
        age = prompt("vehicle_age_years: ", float, minimum=1)
        
        print("\nObservation:")
        print("Mileage does not significantly improve the prediction.")

        print("\nExplanation:")
        print("The other features already provide enough information.")

        vehicle = pd.DataFrame(
            [[engine, horsepower, age, mileage]],
            columns=[
                "engine_size_l",
                "horsepower",
                "vehicle_age_years"
            ]
        )

        prediction = model_a.predict(vehicle)[0]

    print("\nPredicted Fuel Efficiency:", round(prediction, 2), "km/L")


def main():
    while True:
        print(" Fuel Efficiency Menu")
        print("3. Problem 3 - Predict Vehicle Fuel Efficiency")
        print("4. Problem 4 - Measure the Value of Mileage Data")
        print("5. Exit")

        choice = input("Enter your choice (3-4): ")

        if choice == "3":
            problem3()
            input("Press enter key to continue...")
            clear()
        elif choice == "4":
            problem4()
            input("Press enter key to continue...")
            clear()
        elif choice == "5":
            print("Exiting the program...")
            break
        else:
            print("Invalid menu choice.")

if __name__ == "__main__":
    main()
    

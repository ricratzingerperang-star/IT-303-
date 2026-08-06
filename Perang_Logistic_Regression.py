from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

DATASET_PATH = Path(__file__).with_name('student_pass_fail.csv')

FEATURES = [
    'attendance_percent',
    'study_hours_per_week',
    'assignment_average',
    'quiz_average'
]

def main():
    df = pd.read_csv(DATASET_PATH)
    
    X = df[FEATURES]
    y = df['pass_label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42)
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)

    new_student = pd.DataFrame({
        'attendance_percent' : [85],
        'study_hours_per_week': [10],
        'assignment_average': [90],
        'quiz_average': [80]
    })    
    
    pred_label = model.predict(new_student)[0]
    pass_prob = model.predict_proba(new_student)[0][1]
    
    pred_result = "Pass" if pred_label == 1 else "Fail"
    
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Predicted label: {pred_result}")
    print(f"Probability of passing: {pass_prob:.2%}")
    
if __name__ =='__main__':
    main()
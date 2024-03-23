import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, chi2

# Set the color scheme
COLOR_SCHEME = {
    'background': '#f4f4f4',
    'text': '#000000',
    'header': '#2f4f4f',
    'button': '#008080',
}


st.set_page_config(
    page_title="Bankruptcy Prediction App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS styling for the app
st.markdown(
    f"""
    <style>
        .reportview-container {{
            background-color: {COLOR_SCHEME['background']};
            color: {COLOR_SCHEME['text']};
        }}
        .sidebar .sidebar-content {{
            background-color: {COLOR_SCHEME['header']};
            color: {COLOR_SCHEME['text']};
        }}
        .streamlit-button {{
            background-color: {COLOR_SCHEME['button']};
            color: {COLOR_SCHEME['text']};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def feature_engineering(data):
    # Add your feature engineering steps here
    # Example: data['new_feature'] = data['feature1'] * data['feature2']
    return data

def load_data():
    data = pd.read_csv(r"C:\Users\Gaurav\OneDrive\Desktop\Bankruptcy Prediction App\Bankruptcy Prediction.csv")

    # Remove outliers from numerical columns
    numerical_columns = data.select_dtypes(include='float').columns
    for col in numerical_columns:
        data = remove_outliers(data, col)

    # Perform feature engineering
    data = feature_engineering(data)

    return data

def train_model(X_train, y_train, X_test, y_test):
    model_rf = RandomForestClassifier()
    model_rf.fit(X_train, y_train)
    return model_rf

def main():
    # Set the title of your Streamlit app
    st.title('Bankruptcy Prediction App')

    # Load data
    df = load_data()

    # Display the target variable
    st.write("Target Variable:", "Bankrupt?")

    # Replace 'target_column' with the actual target variable in your dataset
    target_column = 'Bankrupt?'

    if target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found in the dataset.")
        st.stop()

    # Assuming 'target_column' is the target variable
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Use SelectKBest to extract top 30 best features
    best_features_selector = SelectKBest(score_func=chi2, k=30)
    fit = best_features_selector.fit(X, y)

    # Get the top 30 features
    selected_features = X.columns[fit.get_support()]

    # Select only the top 30 features
    X_selected = X[selected_features]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train, X_test, y_test)

    # Display model evaluation metrics
    st.subheader("Model Evaluation Metrics:")
    st.write(f"Training Accuracy: {accuracy_score(y_train, model.predict(X_train)):.2%}")
    st.write(f"Testing Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2%}")
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, model.predict(X_test)))
    st.write("Classification Report:")
    st.write(classification_report(y_test, model.predict(X_test)))

    # Add user input for prediction
    st.sidebar.header('User Input:')

    features = {}
    for feature in selected_features:
        features[feature] = st.number_input(feature, min_value=df[feature].min(), max_value=df[feature].max())

    submitted = st.sidebar.button("Predict")

    # Check if the button is clicked
    if submitted:
        # Convert user input to a DataFrame
        user_input = pd.DataFrame([features])

        # Predict bankruptcy on user input
        prediction = model.predict(user_input)
        probability = model.predict_proba(user_input)[:, 1]

        # Display prediction result
        st.subheader('Prediction Result:')
        st.write(f"The model predicts that the company is {'Bankrupt' if prediction[0] == 1 else 'Not Bankrupt'}")
        st.write(f"Probability of Bankruptcy: {probability[0]:.2%}")

if __name__ == '__main__':
    main()

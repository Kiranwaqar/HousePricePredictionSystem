import pandas as pd
import gradio as gr
import traceback
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def process_csv(file):
    try:
        # Load file
        df = pd.read_csv(file, sep='\\s+', header=None)

        df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

        # Fill missing
        df.fillna(df.mean(numeric_only=True), inplace=True)

        # Feature & target split
        X = df.drop('MEDV', axis=1)
        y = df['MEDV']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict & evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Coefficients as DataFrame
        coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient']).reset_index()
        coeff_df.rename(columns={'index': 'Feature'}, inplace=True)

        summary = f"""\
            • Mean Squared Error: {mse:.2f}
            • R-squared: {r2:.2f}"""

        return summary, coeff_df

    except Exception as e:
        return f"An error occurred:\n\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}", None

# Gradio Interface
iface = gr.Interface(
    fn=process_csv,
    inputs=gr.File(label="Upload CSV"),
    outputs=[
        gr.Textbox(label="Model Evaluation"),
        gr.Dataframe(label="Model Coefficients")
    ],
    title="Linear Regression for Housing Data",
    description="Upload your whitespace-separated housing dataset. This app will preprocess it, train a model, and show results in a modern layout."
)

iface.launch()

#  House Price Prediction with Linear Regression

This project demonstrates a simple machine learning pipeline using **Linear Regression** to predict housing prices based on various features. The application also includes a user-friendly **Gradio interface** for uploading a dataset, preprocessing it, training the model, and viewing evaluation results along with model coefficients.

---

##  Key Features

Upload your own dataset (.csv)

Automatic preprocessing:

Handling missing values

Feature scaling with StandardScaler

Train a Linear Regression model

Evaluate performance using:

Mean Squared Error (MSE)

R-squared (R²)

Display model coefficients in a clean, modern table

Easy-to-use Gradio interface

## How to Run
1. Clone the repository
```bash
git clone https://github.com/Kiranwaqar/HousePricePredictionSystem.git
cd Level_1_Task_2
```
2. Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # on Windows
# or
source venv/bin/activate  # on Linux/macOS
```
3. Install the required packages
```bash
pip install -r requirements.txt
```
    If you don’t have a requirements.txt, install manually:
```bash
pip install pandas numpy scikit-learn gradio
```
## Run the Gradio App
```bash
python app.py
```
Then, open the browser at the given local address (e.g. http://127.0.0.1:7860) and interact with the model by uploading your CSV.

## Demo

https://github.com/user-attachments/assets/fd020909-fb27-44b0-a99d-d6bd7e1319c2



Model Used:

 Linear Regression from scikit-learn

Metrics:

 Mean Squared Error (MSE)

 R-squared (R²)

Scaler:

 StandardScaler to normalize features

##   Future Enhancements
Add support for CSVs with different delimiters

Show live prediction by entering custom input

Visualize feature importance with charts

Deploy to Hugging Face Spaces or Streamlit

## Author
Kiran Waqar

Second-year Software Engineering student with a passion for machine learning and education!

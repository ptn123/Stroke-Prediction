# Stroke Prediction

This project aims to predict the likelihood of a stroke using machine learning techniques. The model is built using logistic regression and is trained on a dataset with various health features.

## Project Overview

The stroke prediction model is designed to help identify individuals at risk of having a stroke based on several health indicators. The model leverages machine learning algorithms to provide accurate predictions and is intended for use in health-related applications.

## Features

- **Logistic Regression**: Used for classification to predict the likelihood of stroke.
- **Data Preprocessing**: Includes handling of missing values, encoding categorical variables, and feature scaling.
- **Model Evaluation**: ROC AUC score and other metrics are used to evaluate model performance.

## Dataset

The dataset used for this project includes the following columns:
- `id`: Unique identifier for each record
- `gender`: Gender of the individual
- `age`: Age of the individual
- `hypertension`: Whether the individual has hypertension (0 or 1)
- `heart_disease`: Whether the individual has heart disease (0 or 1)
- `ever_married`: Marital status (Yes or No)
- `work_type`: Type of work the individual does
- `Residence_type`: Type of residence (Urban or Rural)
- `avg_glucose_level`: Average glucose level
- `bmi`: Body Mass Index
- `smoking_status`: Smoking status (formerly smoked, never smoked, or smoke)
- `stroke`: Target variable (0 or 1)

## Installation

To get started with this project, you need to set up your environment and install the required dependencies. Follow these steps:

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/ptn123/Stroke-Prediction.git
    cd Stroke-Prediction
    ```

2. **Create a Virtual Environment** (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**:

    Create a `requirements.txt` file if not already available, and list all dependencies. For example:

    ```plaintext
    pandas
    numpy
    scikit-learn
    joblib
    streamlit
    ```

    Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Model**:

    To use the trained model, ensure you have the `stroke_prediction_model.pkl` file. You can run your model with the following command:

    ```python
    python predict.py
    ```

2. **Streamlit Application**:

    To run the Streamlit application for predictions:

    ```bash
    streamlit run app.py
    ```

    Open your browser and navigate to `http://localhost:8501` to interact with the app.

## Model Deployment

The model has been deployed using [Streamlit](https://streamlit.io/). For deployment details, refer to the [Streamlit documentation](https://docs.streamlit.io/).

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [GitHub](https://github.com) for version control and repository hosting.
- [Scikit-learn](https://scikit-learn.org/) for machine learning algorithms.
- [Streamlit](https://streamlit.io/) for building the web application.


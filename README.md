# Mental Wellness Analysis and Support Strategy Design for the Tech Workforce

## 📝 Project Summary

This capstone project delves into the mental health landscape of the tech industry. The primary goal is to analyze factors influencing the mental well-being of tech professionals and to predict their likelihood of seeking treatment. The project employs a multi-faceted approach, combining Exploratory Data Analysis (EDA), predictive modeling, and customer segmentation to derive actionable insights.

The key components of this analysis include:
* **Exploratory Data Analysis (EDA):** Univariate, bivariate, and multivariate analyses were performed to uncover initial patterns, correlations, and insights within the dataset.
* **Data Preprocessing:** A robust pipeline was constructed using `ColumnTransformer` to handle feature engineering, scaling (`StandardScaler`, `RobustScaler`), and encoding (`OneHotEncoder`) to prepare the data for modeling.
* **Classification:** Supervised learning models, specifically **Random Forest** and **XGBoost classifiers**, were trained to predict whether an individual in the tech workforce would seek mental health treatment.
* **Regression:** **Random Forest** and **XGBoost regressors** were used to predict the age of the respondents based on their survey answers.
* **Clustering:** Unsupervised learning was used to segment the workforce into distinct groups. Dimensionality reduction techniques like **PCA, t-SNE, KernelPCA, and UMAP** were evaluated, with UMAP providing the best separation. Subsequently, **K-Means** and **Agglomerative Clustering** were applied to identify and profile different employee archetypes based on their mental health attitudes and workplace environment.

---

## ⚙️ Setup Instructions

To run this project locally, please follow these steps:

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Amrinder2006/Mental-Wellness-Analysis-and-Support-Strategy-Design-for-the-Tech-Workforce-Analysis.git
    cd Mental-Wellness-Analysis-and-Support-Strategy-Design-for-the-Tech-Workforce-Analysis
    ```

2.  **Create and Activate a Virtual Environment**
    * **For Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * **For macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install Dependencies**
    A `requirements.txt` file should be in the root directory. Install the necessary libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *If you don't have a `requirements.txt` file, you can create one with the following contents:*
    ```
    numpy
    pandas
    matplotlib
    seaborn
    scikit-learn
    xgboost
    umap-learn
    streamlit
    joblib
    ```

4.  **Run the Streamlit Application**
    Once the dependencies are installed, you can launch the interactive web application:
    ```bash
    streamlit run app.py
    ```

---

## 📊 Feature Description

The dataset contains the following features from a 2014 survey on mental health in the tech industry:

| Feature | Description |
| :--- | :--- |
| **Timestamp** | The date and time the survey was submitted. |
| **Age** | The respondent's age. |
| **Gender** | The respondent's gender. |
| **Country** | The country where the respondent lives. |
| **state** | The US state where the respondent lives (if applicable). |
| **self_employed** | Is the respondent self-employed? |
| **family_history** | Does the respondent have a family history of mental illness? |
| **treatment** | Has the respondent sought treatment for a mental health condition? (Target Variable) |
| **work_interfere** | If they have a mental health condition, do they feel it interferes with their work? |
| **no_employees** | The number of employees in the respondent's company. |
| **remote_work** | Does the respondent work remotely at least 50% of the time? |
| **tech_company** | Is the employer primarily a tech company/organization? |
| **benefits** | Does the employer provide mental health benefits? |
| **care_options** | Does the respondent know the options for mental health care their employer provides? |
| **wellness_program** | Has the employer ever discussed mental health as part of an employee wellness program? |
| **seek_help** | Does the employer provide resources to learn more about mental health issues and how to seek help? |
| **anonymity** | Is anonymity protected if the respondent chooses to take advantage of mental health resources? |
| **leave** | How easy is it for the respondent to take medical leave for a mental health condition? |
| **mental_health_consequence** | Does the respondent think that discussing a mental health issue with their employer would have negative consequences? |
| **phys_health_consequence** | Does the respondent think that discussing a physical health issue with their employer would have negative consequences? |
| **coworkers** | Would the respondent be willing to discuss a mental health issue with their coworkers? |
| **supervisor** | Would the respondent be willing to discuss a mental health issue with their direct supervisor(s)? |
| **mental_health_interview** | Would the respondent bring up a mental health issue with a potential employer in an interview? |
| **phys_health_interview** | Would the respondent bring up a physical health issue with a potential employer in an interview? |
| **mental_vs_physical** | Does the respondent feel that their employer takes mental health as seriously as physical health? |
| **obs_consequence** | Has the respondent heard of or observed negative consequences for coworkers with mental health conditions in their workplace? |
| **comments** | Any additional notes or comments from the respondent. |

---

## 🚀 Deployed Streamlit App

You can view and interact with the live project dashboard here:

[**Mental Wellness in Tech Dashboard**](<YOUR_STREAMLIT_APP_LINK>)

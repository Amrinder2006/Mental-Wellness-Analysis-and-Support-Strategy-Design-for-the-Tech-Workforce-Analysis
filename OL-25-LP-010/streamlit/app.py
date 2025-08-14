import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,RobustScaler,OneHotEncoder,MinMaxScaler,MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit,GridSearchCV,RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score,r2_score,classification_report,confusion_matrix,mean_absolute_error,mean_squared_error,roc_auc_score
from sklearn.metrics import silhouette_score
import joblib
from sklearn.decomposition import PCA,KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans,AgglomerativeClustering
import umap.umap_ as umap
import streamlit as st
import os

def Home():
    st.title("Mental Wellness in the Tech Workforce: Analysis and Strategy")
    st.markdown("---")

    st.header("📖 About the Project")
    st.markdown("""
    This project analyzes a 2014 survey dataset on mental health in the tech industry. The primary goals are to understand the factors influencing an employee's decision to seek mental health treatment and to identify distinct employee archetypes through clustering.
    
    By leveraging machine learning, we aim to build predictive models and derive actionable insights that can help companies design better mental wellness support strategies.
    """)
    st.markdown("---")

    st.header("📊 The Dataset")
    st.markdown("""
    The dataset contains responses from tech employees, covering a range of topics:
    - **Demographics:** Age, Gender, Country, State.
    - **Workplace Environment:** Company size, remote work status, tech company status.
    - **Mental Health Attitudes & Benefits:** Availability of benefits, wellness programs, anonymity, and perceived consequences of discussing mental health.
    - **Target Variable (Classification):** `treatment` - Whether the employee has sought treatment for a mental health condition.
    """)
    st.markdown("---")

    st.header("🤔 Problems Faced")
    st.markdown("""
    Several challenges were addressed during the project:
    - **Data Cleaning:** The dataset contained inconsistencies, missing values, and outliers (e.g., in the 'Age' column) that required careful cleaning and imputation.
    - **Feature Engineering:** Many features were categorical. Transforming them into a numerical format suitable for machine learning models using techniques like One-Hot Encoding was crucial.
    - **High Cardinality:** Features like 'Country' and 'state' had many unique values, which can make modeling difficult. These were handled strategically during preprocessing.
    - **Complex Interactions:** Understanding the nuanced relationships between workplace culture, benefits, and an individual's willingness to seek help required detailed exploratory analysis.
    """)
    st.markdown("---")

    st.header("🛠️ Work Done")
    st.markdown("""
    The project was structured into several key phases:

    **1. Exploratory Data Analysis (EDA):**
       - Performed **univariate, bivariate, and multivariate analysis** to understand data distributions and relationships between features.
       - Visualized key insights using `seaborn` and `matplotlib`.

    **2. Data Preprocessing & Pipelines:**
       - Constructed a robust preprocessing pipeline using `sklearn.pipeline.Pipeline` and `ColumnTransformer`.
       - Applied different scalers (**StandardScaler, RobustScaler**) and encoders (**OneHotEncoder**) to the appropriate columns.

    **3. Predictive Modeling (Supervised Learning):**
       - **Classification:** Trained `RandomForestClassifier` and `XGBClassifier` to predict if an employee would seek `treatment`.
       - **Regression:** Trained `RandomForestRegressor` and `XGBRegressor` to predict the `Age` of a respondent.

    **4. Customer Segmentation (Unsupervised Learning):**
       - **Dimensionality Reduction:** Evaluated **PCA, t-SNE, KernelPCA, and UMAP**, with UMAP providing the best results for visualizing clusters.
       - **Clustering:** Applied **K-Means** and **Agglomerative Clustering** on the dimensionally-reduced data to segment employees into distinct groups.
       - **Cluster Profiling:** Analyzed the characteristics of each cluster to provide meaningful business insights.
    """)

def EDA():

    st.header("Required Imports")
    st.markdown("""
    ```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns""")
    st.divider()
    st.header("Dataset")
    url='OL-25-LP-010/streamlit/survey.csv'
    current_dir = os.path.dirname(os.path.abspath(__file__))

# Join with the CSV file name
    csv_path = os.path.join(current_dir, "survey.csv")

# Read CSV
    df = pd.read_csv(csv_path)
    st.dataframe(df)
    st.write(f"Rows={df.shape[0]},Columns={df.shape[1]}")
    st.divider()
    st.header("Data Insights")
    cols=st.columns(2)
    with cols[0]:
        st.subheader("Columns")
        for i in range(0,len(df.columns)-1,4):
            subcols=st.columns(4)
            with subcols[0]:
                st.code(df.columns[i])
            with subcols[1]:
               if(i+2<27):
                st.code(df.columns[i+1])
            with subcols[2]:
               if(i+1<27):
                st.code(df.columns[i+2])
            with subcols[3]:
               if(i+3<27):
                st.code(df.columns[i+3])
        
    with cols[1]:
        st.subheader("Data types of columns")
        st.dataframe(df.dtypes)
    st.divider()
    cols1=st.columns(2)
    with cols1[0]:
        st.subheader("Description of Age in Data")
        st.dataframe(df.describe())
    with cols1[1]:
       st.subheader("Missing Values")
       st.code(df.isna().sum())
    df=df.fillna('No_data')
    df.drop(['Timestamp','comments','state'],axis=1,inplace=True)
    df=df[(df['Age']>0)  & (df['Age']<100)]
    df['Gender']=df['Gender'].replace({'male':'Male','m':'Male','M':'Male','female':'Female','f':'Female','F':'Female'})
    df=df[df['Gender'].isin(['Male','Female'])]
    value=df['Country'].value_counts().head(7).index
    df['Country']=df['Country'].apply(lambda X:X if X in value else 'Other')
    st.divider()
    st.header("Data Cleaning")
    st.markdown("""
    ```python 
    df=df.fillna('No_data')
    df.drop(['Timestamp','comments','state'],axis=1,inplace=True)
    df=df[(df['Age']>0)  & (df['Age']<100)]
    df['Gender']=df['Gender'].replace({'male':'Male','m':'Male','M':'Male','female':'Female','f':'Female','F':'Female'})
    df=df[df['Gender'].isin(['Male','Female'])]
    value=df['Country'].value_counts().head(7).index
    df['Country']=df['Country'].apply(lambda X:X if X in value else 'Other')""")
    st.divider()
    st.header("Data Visualization")
    st.subheader("Univariate Analysis")
    fig,axs=plt.subplots(4,6,figsize=(30,30))
    ax=axs.flatten()
    for i,ax in enumerate(ax):
        if(i==23):
            sns.boxplot(y=df['Age'],ax=ax)
            ax.set_xlabel("Age")
            break
        sns.countplot(x=df[df.columns[i+1]],ax=ax)
        ax.set_xlabel(f"{df.columns[i+1]}")
    plt.suptitle("Countplot of Different Features",fontsize=40,y=0.92)
    st.pyplot(fig)
    st.divider()
    fig1,ax=plt.subplots()
    plt.pie(x=df['treatment'].value_counts().values,labels=['Yes', 'No'],autopct='%.2f%%')
    plt.title("Pie Chart of distribution of Treatment values")
    st.pyplot(fig1)
    st.divider()
    fig2,ax=plt.subplots()
    sns.histplot(data=df,x='Age',kde=True)
    plt.title("Histplot of different ages")
    st.pyplot(fig2)
    st.divider()
    st.subheader("Bivariate Analysis")
    fig1,axs=plt.subplots(4,6,figsize=(30,30))
    ax=axs.flatten()
    for i,ax in enumerate(ax):
        if(i==23):
            sns.boxplot(y=df['Age'],x=df['treatment'],ax=ax)
            ax.set_xlabel("treatment")
            ax.set_ylabel("Age")
            break
        sns.countplot(x=df[df.columns[i+1]],hue=df['treatment'],ax=ax)
        ax.set_xlabel(f"{df.columns[i+1]}")
    plt.suptitle("Countplot of Different Features with treatment status",fontsize=40,y=0.92)
    st.pyplot(fig1)
    st.divider()
    fig4,ax=plt.subplots()
    sns.countplot(data=df,x='coworkers',hue='supervisor')
    plt.title("Sharing mental health issue with coworkers and supervisors")
    st.pyplot(fig4)
    st.divider()
    st.subheader("Multivariate Analysis")
    df['treatment']=df['treatment'].map({'Yes':1,'No':0})
    df1=df.groupby('Country')[['treatment','Age']].mean().sort_values('treatment',ascending=False)
    index=df1.index
    df1['treatment']=df1['treatment'].round(2)
    st.subheader("Data Grouping")
    st.markdown("""
```python
    df['treatment']=df['treatment'].map({'Yes':1,'No':0})
    df1=df.groupby('Country')[['treatment','Age']].mean().sort_values('treatment',ascending=False)
    index=df1.index
    df1['treatment']=df1['treatment'].round(2)""")
    st.divider()
    st.dataframe(df1)

    fig3=plt.figure(figsize=(12,8))
    sns.barplot(data=df1,x='treatment',y='Age',hue=index)
    plt.title("Mental Health Treatment Rate by Countries with their respective mean ages",fontsize=20)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig3)
    st.divider()
    fig6=plt.figure(figsize=(12,20))
    g=sns.catplot(data=df,x='care_options',hue='treatment',col='benefits',kind='count')
    st.pyplot(g.fig)
    st.header("Information from Dataset")
    st.divider()
    fig,ax=plt.subplots()
    st.subheader("More ratio of Females seek treatment for a mental health condition than Men.")
    sns.countplot(x=df['Gender'],hue=df['treatment'])
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.title("Genderwise Treatment status")
    st.pyplot(fig)
    st.divider()
    st.subheader("More people with family history of mental illness seek treatment than with no history.")
    fig,ax=plt.subplots()
    sns.countplot(data=df,x='family_history',hue='treatment')
    plt.xlabel("Family History")
    plt.ylabel("Count")
    plt.title("Family History Correlation with Treatment")
    st.pyplot(fig)
    st.divider()
    st.subheader("People with knowledge of benefits and care options seek more treatment.")
    sns.catplot(data=df,x='care_options',hue='treatment',col='benefits',kind='count')
    st.pyplot(g.fig)
    st.divider()
    st.subheader("Most of the people who discuss about their mental health issue with their coworkers are also comfortable discussing it with their supervisors.")
    fig,ax=plt.subplots()
    sns.countplot(data=df,x='coworkers',hue='supervisor')
    plt.title("Sharing mental health issue with coworkers and supervisors")
    st.pyplot(fig)
    st.divider()
    st.subheader("Age has many outliers and has a Non-Gaussian Curve.")
    fig,axs=plt.subplots(1,2,figsize=(12,6))
    ax=axs.flatten()
    sns.boxplot(data=df,y='Age',ax=ax[0])
    ax[0].set_ylabel("Age")
    ax[0].set_title("Boxplot of Age")
    sns.histplot(data=df,x='Age',kde=True,ax=ax[1])
    ax[1].set_title("Histplot of different ages")
    st.pyplot(fig)
    st.divider()
    st.subheader("Australia has highest treatment mean with lowest mean of ages while US has second highest treatment mean with highest mean of ages.So age doesn't seem to be a major factor for treatment.")
    fig=plt.figure(figsize=(12,8))
    sns.barplot(data=df1,x='treatment',y='Age',hue=index)
    plt.title("Mental Health Treatment Rate by Countries with their respective mean ages")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)
    st.divider()

def Classification():
   st.title("Classification")
   st.header("Required Imports")
   st.markdown("""
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,RobustScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix""")
   st.header("Dataset")
   st.text("For classification,we will be taking the cleaned dataset from the EDA section.")
   df=pd.read_csv('OL-25-LP-010/streamlit/df_cleaned.csv')
   st.dataframe(df)
   st.text(f"Rows={df.shape[0]},Columns={df.shape[1]}")
   df.drop('Unnamed: 0',axis=1,inplace=True)
   st.header("Data Insights")
   cols=st.columns(3)
   with cols[0]:
      st.subheader("Description of data")
      st.dataframe(df.describe())
   with cols[1]:
      st.subheader("Data Types of Features")
      st.code(df.dtypes)
   with cols[2]:
      st.subheader("Missing Values")
      st.code(df.isna().sum())
   df['leave']=df['leave'].map({'Very difficult': 0,'Somewhat difficult': 1,"Don't know": 2,'Somewhat easy': 3,'Very easy': 4})
   df['risk_score']= df['family_history'].map({'Yes':1,'No':0})+df['mental_health_consequence'].map({'Yes':2,'No':0,"Don't know":1})+df['mental_vs_physical'].map({'Yes':2,'No':0,"Don't know":1})
   df['Support']=df['benefits'].map({'Yes':1,'No':0,"Don't know":0.5})+df['care_options'].map({'Yes':1,'No':0,"Not sure":0.5})+df['wellness_program'].map({'Yes':1,'No':0,"Don't know":0.5})+df['leave']
   df['Age']=np.log(df['Age'])
   st.divider()
   st.markdown("""
## Feature Engineering
```python
df.drop('Unnamed: 0',axis=1,inplace=True)
df['leave']=df['leave'].map({'Very difficult': 0,'Somewhat difficult': 1,"Don't know": 2,'Somewhat easy': 3,'Very easy': 4})
df['risk_score']= df['family_history'].map({'Yes':1,'No':0})+df['mental_health_consequence'].map({'Yes':2,'No':0,"Don't know":1})+df['mental_vs_physical'].map({'Yes':2,'No':0,"Don't know":1})
df['Support']=df['benefits'].map({'Yes':1,'No':0,"Don't know":0.5})+df['care_options'].map({'Yes':1,'No':0,"Not sure":0.5})+df['wellness_program'].map({'Yes':1,'No':0,"Don't know":0.5})+df['leave']
df['Age']=np.log(df['Age'])
""")
   st.divider()
   fig=plt.figure()
   sns.heatmap(df[['risk_score','Support','Age']].corr(),annot=True,cmap='coolwarm',fmt='.2f')
   st.pyplot(fig)
   st.divider()
   x=df.drop('treatment',axis=1)
   y=df['treatment']
   sss=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=40)
   for train_idx, test_idx in sss.split(x, y):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

   st.divider()
   st.title("RandomForests Classifier Results")
   st.header("Pipeline")
   encode_cols=list(df.select_dtypes(include='object').columns)
   
   pipeline=Pipeline([
    ("preprocess",ColumnTransformer([
        ('encode',OneHotEncoder(handle_unknown='ignore'),encode_cols),
        ('Robust',RobustScaler(),['Age'])
        ]),
    ),
    ('Train',RandomForestClassifier(n_estimators=400,min_samples_leaf=1,max_depth=6,max_features='sqrt',min_samples_split=3,n_jobs=-1,criterion='gini',random_state=42))
    ])
   st.code(pipeline)
   pipeline.fit(x_train,y_train)
   y_pred=pipeline.predict(x_test)
   st.divider()
   st.header("Evaluation")
   st.text(f"Accuracy:{accuracy_score(y_test,y_pred)}")
   fig1=plt.figure()
   sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap='coolwarm',fmt='d')
   plt.xlabel("Predicted Values")
   plt.ylabel("Real Values")
   plt.title("Confusion Matrix")
   st.pyplot(fig1)
   st.code(classification_report(y_test,y_pred))
   st.divider()
   st.subheader("Important Features")
   importances=pipeline.named_steps['Train'].feature_importances_
   sorted_importance=np.argsort(importances)[::-1]
   sorted_importance=sorted_importance[0:10]

   feature = pipeline.named_steps['preprocess'].get_feature_names_out()
   fig,ax=plt.subplots()
   sns.barplot(y=feature[sorted_importance],x=importances[sorted_importance])
   plt.xlabel("Feature Coefficients")
   plt.ylabel("Feature Names")
   plt.title("Top 10 important Features")
   st.pyplot(fig)
   st.title("XGB Classifier Results")
   st.header("Pipeline")
   encode_cols=list(df.select_dtypes(include='object').columns)
   pipeline=Pipeline([
    ("preprocess",ColumnTransformer([
        ('encode',OneHotEncoder(handle_unknown='ignore'),encode_cols),
        ('Robust',RobustScaler(),['Age'])
        ]),
    ),
    ('Train',XGBClassifier(n_estmimators=400,max_depth=5,subsample=0.62,max_features='log2'))
    ])
   pipeline.fit(x_train,y_train)
   y_pred=pipeline.predict(x_test)
   st.code(pipeline)
   st.divider()
   st.header("Evaluation")
   st.text(f"Accuracy:{accuracy_score(y_test,y_pred)}")
   fig1=plt.figure()
   sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap='coolwarm',fmt='d')
   plt.xlabel("Predicted Values")
   plt.ylabel("Real Values")
   plt.title("Confusion Matrix")
   st.pyplot(fig1)
   st.code(classification_report(y_test,y_pred))
   st.divider()
   st.subheader("Important Features")
   importances=pipeline.named_steps['Train'].feature_importances_
   sorted_importance=np.argsort(importances)[::-1]
   sorted_importance=sorted_importance[0:10]

   feature = pipeline.named_steps['preprocess'].get_feature_names_out()
   fig,ax=plt.subplots()
   sns.barplot(y=feature[sorted_importance],x=importances[sorted_importance])
   plt.xlabel("Feature Coefficients")
   plt.ylabel("Feature Names")
   plt.title("Top 10 important Features")
   st.pyplot(fig)
   st.divider()
   st.subheader("From the results,we can see that the performance of Random Forests Classifier is better.So we will be taking that model for prediction.")
   st.divider()
   st.title("Prediction")
   feat=[]
   st.header("Features")
   age=st.number_input("Age",min_value=1,max_value=100,step=1,value=28)
   feat.append(age)
   risk_score=0
   support=0
   for i in df.columns[1:-2]:
      if i=='treatment':
         continue
      elif i in ['family_history','mental_health_consequence','mental_vs_physical']:
         j=st.selectbox(f"{i}",options=df[i].unique(),index=0)
         if(j=='Yes'):
            risk_score+=1
         elif(j=="Don't know"):
            risk_score+=0.5
         else:
            risk_score+=0
         feat.append(j)
      elif i in ['benefits','care_options','wellness_program']:
         j=st.selectbox(f"{i}",options=df[i].unique(),index=0)
         if(j=='Yes'):
            support+=1
         elif(j=="Don't know"):
            support+=0.5
         elif(j=="Not sure"):
            support+=0.5
         else:
            support+=0
         feat.append(j)
         
      else:
         j=st.selectbox(f"{i}",options=df[i].unique(),index=0)
         feat.append(j)
   feat.append(risk_score)
   feat.append(support)
   st.code(feat)
   feat=np.array(feat)
   feat=feat.reshape(1,-1)
   st.header("Hyperparameters")
   cols=st.columns(3)
   with cols[0]:
      n_estmimators=st.number_input("n_estimators",value=350,step=50)
      min_samples_leaf=st.slider("Min_samples_leaf",min_value=1,max_value=20,step=1,value=1)
   with cols[1]:
      max_depth=st.number_input("Max_depth",value=6,step=1)
      max_features=st.selectbox('Max_features',options=['sqrt','log2',None]) or 'sqrt'
   with cols[2]:
      min_samples_split=st.number_input('min_samples_split',value=3,step=1)
      criterion=st.radio('Criterion',options=['gini','entropy']) or 'gini'
   pipeline=Pipeline([
    ("preprocess",ColumnTransformer([
        ('encode',OneHotEncoder(handle_unknown='ignore'),encode_cols),
        ('Robust',RobustScaler(),['Age'])
        ]),
    ),
    ('Train',RandomForestClassifier(n_estimators=n_estmimators,min_samples_leaf=min_samples_leaf,max_depth=max_depth,max_features=max_features,min_samples_split=min_samples_split,n_jobs=-1,criterion=criterion,random_state=42))
    ])
   st.code(pipeline)
   if st.button("Predict"):
      pipeline.fit(x,y)
      y_pred=pipeline.predict(pd.DataFrame(data=feat,columns=x.columns))
      if(y_pred==1):
         st.code(y_pred)
         st.code("The Person will likely seek Treatment")
      else:
         st.code(y_pred)
         st.code("The Person will not likely seek Treatment")

def Regression():
   st.title("Regression")
   st.header("Required Imports")
   st.markdown("""
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,RobustScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from scipy.stats import randint, uniform
from xgboost import XGBRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error""")
   st.header("Dataset")
   st.text("For Regression,we will be taking the cleaned dataset from the EDA section.")
   df=pd.read_csv('OL-25-LP-010/streamlit/df_cleaned.csv')
   st.dataframe(df)
   st.text(f"Rows={df.shape[0]},Columns={df.shape[1]}")
   df.drop('Unnamed: 0',axis=1,inplace=True)
   st.header("Data Insights")
   cols=st.columns(3)
   with cols[0]:
      st.subheader("Description of data")
      st.dataframe(df.describe())
   with cols[1]:
      st.subheader("Data Types of Features")
      st.code(df.dtypes)
   with cols[2]:
      st.subheader("Missing Values")
      st.code(df.isna().sum())
   df['leave']=df['leave'].map({'Very difficult': 0,'Somewhat difficult': 1,"Don't know": 2,'Somewhat easy': 3,'Very easy': 4})
   df['risk_score']= df['family_history'].map({'Yes':1,'No':0})+df['mental_health_consequence'].map({'Yes':2,'No':0,"Don't know":1})+df['mental_vs_physical'].map({'Yes':2,'No':0,"Don't know":1})
   df['Support']=df['benefits'].map({'Yes':1,'No':0,"Don't know":0.5})+df['care_options'].map({'Yes':1,'No':0,"Not sure":0.5})+df['wellness_program'].map({'Yes':1,'No':0,"Don't know":0.5})+df['leave']
   df['work_interfere']=df['work_interfere'].map({'Never': 0,'Rarely': 1,'Sometimes': 3,'Often': 4,'No_data': 2})
   df['no_employees']=df['no_employees'].map({'1-5': 0,'6-25': 1,'26-100': 2,'100-500': 3,'500-1000': 4,'More than 1000': 5})
   st.divider()
   st.markdown("""
## Feature Engineering
```python
df.drop('Unnamed: 0',axis=1,inplace=True)
df['leave']=df['leave'].map({'Very difficult': 0,'Somewhat difficult': 1,"Don't know": 2,'Somewhat easy': 3,'Very easy': 4})
df['risk_score']= df['family_history'].map({'Yes':1,'No':0})+df['mental_health_consequence'].map({'Yes':2,'No':0,"Don't know":1})+df['mental_vs_physical'].map({'Yes':2,'No':0,"Don't know":1})
df['Support']=df['benefits'].map({'Yes':1,'No':0,"Don't know":0.5})+df['care_options'].map({'Yes':1,'No':0,"Not sure":0.5})+df['wellness_program'].map({'Yes':1,'No':0,"Don't know":0.5})+df['leave']
df['work_interfere']=df['work_interfere'].map({'Never': 0,'Rarely': 1,'Sometimes': 3,'Often': 4,'No_data': 2})
df['no_employees']=df['no_employees'].map({'1-5': 0,'6-25': 1,'26-100': 2,'100-500': 3,'500-1000': 4,'More than 1000': 5})
""")
   st.divider()
   x=df.drop('Age',axis=1)
   y=df['Age']
   x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
   st.title("XGB Regressor Results")

   st.divider()
   st.header("Pipeline")
   encode_cols=list(df.select_dtypes(include='object').columns)
   
   encode_cols=list(df.select_dtypes(include='object').columns)
   pipeline=Pipeline([
    ("preprocess",ColumnTransformer([
        ('encode',OneHotEncoder(handle_unknown='ignore'),encode_cols)
        ]),
    ),
('Train',XGBRegressor(n_estmimators=400,learning_rate=0.07,max_depth=3,gamma=0.6,reg_alpha=0.7,reg_lambda=0.2,subsample=0.80))
])
   st.code(pipeline)
   pipeline.fit(x_train,y_train)
   y_pred=pipeline.predict(x_test)
   st.divider()
   st.header("Evaluation")
   st.text(f"R2 Score is:{r2_score(y_test,y_pred)}")
   st.text(f"Mean Squared Error is:{mean_squared_error(y_test,y_pred)}")
   st.text(f"Root Mean Squared Error is:{mean_squared_error(y_test,y_pred)**(1/2)}")
   st.text(f"Mean Absolute Error is:{mean_absolute_error(y_test,y_pred)}")
   st.divider()
   st.subheader("Important Features")
   importances=pipeline.named_steps['Train'].feature_importances_
   sorted_importance=np.argsort(importances)[::-1]
   sorted_importance=sorted_importance[0:10]

   feature = pipeline.named_steps['preprocess'].get_feature_names_out()
   fig,ax=plt.subplots()
   sns.barplot(y=feature[sorted_importance],x=importances[sorted_importance])
   plt.xlabel("Feature Coefficients")
   plt.ylabel("Feature Names")
   plt.title("Top 10 important Features")
   st.pyplot(fig)
   st.divider()
   st.title("Random Forest Regressor Results")
   st.divider()
   st.header("Pipeline")
   encode_cols=list(df.select_dtypes(include='object').columns)
   pipeline=Pipeline([
    ("preprocess",ColumnTransformer([
        ('encode',OneHotEncoder(handle_unknown='ignore'),encode_cols)
        ]),
    ),
   ('Train',RandomForestRegressor(max_depth=6,max_features='sqrt',min_samples_leaf=1,min_samples_split=9,n_estimators=216,n_jobs=-1))
   ])
   st.code(pipeline)
   st.divider()
   pipeline.fit(x_train,y_train)
   y_pred=pipeline.predict(x_test)
   st.divider()
   st.header("Evaluation")
   st.text(f"R2 Score is:{r2_score(y_test,y_pred)}")
   st.text(f"Mean Squared Error is:{mean_squared_error(y_test,y_pred)}")
   st.text(f"Root Mean Squared Error is:{mean_squared_error(y_test,y_pred)**(1/2)}")
   st.text(f"Mean Absolute Error is:{mean_absolute_error(y_test,y_pred)}")
   st.divider()
   st.subheader("Important Features")
   importances=pipeline.named_steps['Train'].feature_importances_
   sorted_importance=np.argsort(importances)[::-1]
   sorted_importance=sorted_importance[0:10]

   feature = pipeline.named_steps['preprocess'].get_feature_names_out()
   fig,ax=plt.subplots()
   sns.barplot(y=feature[sorted_importance],x=importances[sorted_importance])
   plt.xlabel("Feature Coefficients")
   plt.ylabel("Feature Names")
   plt.title("Top 10 important Features")
   st.pyplot(fig)
   st.divider()
   st.subheader("Since performance of XGB Regressor is better,we will be using that model for prediction")
   st.title("Prediction")
   feat=[]
   st.header("Features")
   risk_score=0
   support=0
   for i in x.columns[:-2]:
      if i in ['family_history','mental_health_consequence','mental_vs_physical']:
         j=st.selectbox(f"{i}",options=df[i].unique(),index=0)
         if(j=='Yes'):
            risk_score+=1
         elif(j=="Don't know"):
            risk_score+=0.5
         else:
            risk_score+=0
         feat.append(j)
      elif i in ['benefits','care_options','wellness_program']:
         j=st.selectbox(f"{i}",options=df[i].unique(),index=0)
         if(j=='Yes'):
            support+=1
         elif(j=="Don't know"):
            support+=0.5
         elif(j=="Not sure"):
            support+=0.5
         else:
            support+=0
         feat.append(j)
         
      else:
         j=st.selectbox(f"{i}",options=df[i].unique(),index=0)
         feat.append(j)
   feat.append(risk_score)
   feat.append(support)
   st.code(feat)
   feat=np.array(feat)
   feat=feat.reshape(1,-1)
   st.header("Hyperparameters")
   cols=st.columns(3)
   with cols[0]:
      n_estmimators=st.number_input("n_estimators",value=400,step=50)
      learning_rate=st.slider("Learning Rate",min_value=0.01,max_value=3.0,step=0.01,value=0.07)
   with cols[1]:
      max_depth=st.number_input("Max_depth",value=3,step=1)
      gamma=st.slider('Gamma',value=0.6,step=0.1,max_value=4.0)
   with cols[2]:
      reg_alpha=st.slider('reg_alpha',value=0.7,step=0.01)
      reg_lambda=st.slider('reg_lambda',value=0.2,step=0.01)
   pipeline=Pipeline([
    ("preprocess",ColumnTransformer([
        ('encode',OneHotEncoder(handle_unknown='ignore'),encode_cols)
        ]),
    ),
    ('Train',XGBRegressor(n_estimators=n_estmimators,learning_rate=learning_rate,max_depth=max_depth,reg_alpha=reg_alpha,reg_lambda=reg_lambda,gamma=gamma,subsample=0.8,random_state=42))
    ])
   st.code(pipeline)
   if st.button("Predict"):
      pipeline.fit(x,y)
      y_pred=pipeline.predict(pd.DataFrame(data=feat,columns=x.columns))
      st.code(f"The Predicted Age of person is {int(y_pred)}")

def Clustering():
    st.title("Clustering")
    st.header("Dataset")
    st.text("For clustering,we will be using the cleaned dataset from the EDA portion")
    df=pd.read_csv('OL-25-LP-010/streamlit/df_cleaned.csv')
    st.dataframe(df)
    st.divider()
    st.header("Data Insights")
    cols=st.columns(3)
    with cols[0]:
      st.subheader("Description of data")
      st.dataframe(df.describe())
    with cols[1]:
      st.subheader("Data Types of Features")
      st.code(df.dtypes)
    with cols[2]:
      st.subheader("Missing Values")
      st.code(df.isna().sum())
    df.drop(['Unnamed: 0','Gender','Country','phys_health_consequence','phys_health_interview','no_employees'],axis=1,inplace=True)
    df['work_interfere']=df['work_interfere'].map({'Never': 0,'Rarely': 1,'Sometimes': 3,'Often': 4,'No_data': 2})
    df['leave']=df['leave'].map({'Very difficult': 0,'Somewhat difficult': 1,"Don't know": 2,'Somewhat easy': 3,'Very easy': 4})
    df['mental_health_consequence']=df['mental_health_consequence'].map({'No': 0,"Maybe": 0.5,'Yes': 1})
    df['mental_vs_physical']=df['mental_vs_physical'].map({'Yes':1,'No':0,"Don't know":0.5})
    df['risk_score']= df['family_history'].map({'Yes':1,'No':0})+df['mental_health_consequence']+df['mental_vs_physical']
    df['Support']=df['benefits'].map({'Yes':1,'No':0,"Don't know":0.5})+df['care_options'].map({'Yes':1,'No':0,"Not sure":0.5})+df['wellness_program'].map({'Yes':1,'No':0,"Don't know":0.5})+df['leave']
    columns=['family_history','benefits','care_options','wellness_program','seek_help','anonymity','coworkers','supervisor','mental_health_interview']
    dict={'Yes':1,'No':0,"Don't know":0.5,'Some of them':0.5,'Maybe':0.5,'Not sure':0.5}
    for i in columns:
      df[i]=df[i].map(dict)
    st.divider()
    st.markdown("""
## Feature Engineering
```python
df.drop(['Unnamed: 0','Gender','Country','phys_health_consequence','phys_health_interview','no_employees'],axis=1,inplace=True)
df['work_interfere']=df['work_interfere'].map({'Never': 0,'Rarely': 1,'Sometimes': 3,'Often': 4,'No_data': 2})
df['leave']=df['leave'].map({'Very difficult': 0,'Somewhat difficult': 1,"Don't know": 2,'Somewhat easy': 3,'Very easy': 4})
df['mental_health_consequence']=df['mental_health_consequence'].map({'No': 0,"Maybe": 0.5,'Yes': 1})
df['mental_vs_physical']=df['mental_vs_physical'].map({'Yes':1,'No':0,"Don't know":0.5})
df['risk_score']= df['family_history'].map({'Yes':1,'No':0})+df['mental_health_consequence']+df['mental_vs_physical']
df['Support']=df['benefits'].map({'Yes':1,'No':0,"Don't know":0.5})+df['care_options'].map({'Yes':1,'No':0,"Not sure":0.5})+df['wellness_program'].map({'Yes':1,'No':0,"Don't know":0.5})+df['leave']  
columns=['family_history','benefits','care_options','wellness_program','seek_help','anonymity','coworkers','supervisor','mental_health_interview']
dict={'Yes':1,'No':0,"Don't know":0.5,'Some of them':0.5,'Maybe':0.5,'Not sure':0.5}
for i in columns:
    df[i]=df[i].map(dict)         
""")
    st.divider()
    st.header("Preprocessor")
    preprocessor=ColumnTransformer([
    ('robust',RobustScaler(),['Age']),
    ('encode',OneHotEncoder(),df.select_dtypes(include='object').columns)
   ],
   remainder='passthrough'
   )
    st.code(preprocessor)
    st.divider()
    pipeline1=Pipeline([
    ('preprocess',preprocessor),
    ('scale',StandardScaler()),
    ('PCA',PCA(n_components=2,random_state=42)),
    ])
    pipeline2=Pipeline([
    ('preprocess',preprocessor),
    ('scale',StandardScaler()),
    ('TSNE',TSNE(n_components=2,random_state=42)),
    # ('scale',StandardScaler())
    ])
    pipeline3=Pipeline([
    ('preprocess',preprocessor),
    ('scale',StandardScaler()),
    ('KernelPCA',KernelPCA(n_components=2,random_state=42)),
    ])
    pipeline4=Pipeline([
    ('preprocess',preprocessor),
    ('scale',StandardScaler()),
    ('UMAP',umap.UMAP(n_components=2,random_state=42)),
    
    ])
    pipeline={'PCA':pipeline1,'TSNE':pipeline2,'KernelPCA':pipeline3,'UMAP':pipeline4}
    st.header("Dimensionality Reduction")
    st.subheader("Pipelines")
    st.code(pipeline1)
    st.code(pipeline2)
    st.code(pipeline3)
    st.code(pipeline4)
    st.divider()
    fig,axs=plt.subplots(2,2,figsize=(20,20))
    ax=axs.flatten()
    for i,(name,model) in enumerate(pipeline.items()):
      df_scaled=model.fit_transform(df)
      sns.scatterplot(x=df_scaled[:,0],y=df_scaled[:,1],color='red',ax=ax[i])
      ax[i].set_xlabel(f"{name}-1")
      ax[i].set_ylabel(f"{name}-2")
      ax[i].set_title(f"{name}")
    plt.suptitle("Dimensionality-Reduction")
    st.pyplot(fig)
    st.divider()
    st.header("Silhouette Score")
    st.subheader("KMeans Clustering")
    np.random.seed(42)
    for i in range(2,10):
      pipeline=Pipeline([
        ('preprocess',preprocessor),
        ('scale',StandardScaler()),
        ('reduce',umap.UMAP(n_components=2,n_jobs=-1))
    ])
      df_reduce=pipeline.fit_transform(df)
      kmeans=KMeans(n_clusters=i,n_init=10,init='k-means++',random_state=41)
      labels=kmeans.fit_predict(df_reduce)
      st.code(f"Silhouette Score for {i} clusters is {silhouette_score(df_reduce,labels)}")
    st.divider()
    st.subheader("Agglomerative Clustering")
    for i in range(2,10):
      pipeline=Pipeline([
        ('preprocess',preprocessor),
        ('scale',StandardScaler()),
        ('reduce',umap.UMAP(n_components=2,n_jobs=-1))
    ])
      reduced_data=pipeline.fit_transform(df)
      agglo=AgglomerativeClustering(n_clusters=i)
      labels=agglo.fit_predict(reduced_data)
      st.code(f"Silhouette Score for {i} clusters is {silhouette_score(reduced_data,labels)}")
    st.divider()
    st.header("Hyperparameters")
    st.subheader("KMeans Clustering")
    init=st.selectbox("init",options=['k-means++','random'],index=0)
    n_init=st.slider("n_init",value=10,max_value=20)
    st.subheader("Agglomerative Clusering")
    linkage=st.selectbox("Linkage",options=['ward','single','complete','average'],index=0)
    if linkage=='ward':
       metric=st.selectbox("Metric",options=['euclidean'])
    else:
       metric=st.selectbox("Metric",options=['euclidean', 'manhattan', 'cosine', 'l1', 'l2'])
    st.divider()
    st.header("KMeans Clustering Results")
    pipeline1=Pipeline([
    ('preprocess',preprocessor),
    ('scale',StandardScaler()),
    ('reduce',umap.UMAP(n_components=2,n_jobs=-1))

])
    joblib.dump(pipeline1,'ml_unsupervise_pipeline')
    fig=plt.figure(figsize=(20,10))
    kmeans=KMeans(n_clusters=5,n_init=n_init,init=init,random_state=40)
    df_reduce=pd.DataFrame(pipeline1.fit_transform(df),columns=['umap-1','umap-2'])
    labels=kmeans.fit_predict(df_reduce)
    df_reduce['Clusters']=labels
    df_reduce
    sns.scatterplot(x=df_reduce.iloc[:,0],y=df_reduce.iloc[:,1],hue=df_reduce['Clusters'],palette='viridis')
    plt.title("Scatterplot of KMeans Clustering of different data points")
    st.pyplot(fig)
    st.divider()
    st.code(f"Inertia:{kmeans.inertia_} \n\n\n Center of clusters:{kmeans.cluster_centers_}")
    st.code(df_reduce['Clusters'].value_counts())
    st.divider()
    st.header("Agglomerative Clustering Results")
    pipeline2=Pipeline([
    ('preprocess',preprocessor),
    ('scale',StandardScaler()),
    ('reduce',umap.UMAP(n_components=2,n_jobs=-1))

])

    fig1=plt.figure(figsize=(20,10))
    agglo=AgglomerativeClustering(n_clusters=5,metric=metric,linkage=linkage)
    df_reduce1=pd.DataFrame(pipeline2.fit_transform(df),columns=['umap-1','umap-2'])
    labels=agglo.fit_predict(df_reduce)
    df_reduce1['Clusters']=labels
    df_reduce1
    sns.scatterplot(x=df_reduce.iloc[:,0],y=df_reduce.iloc[:,1],hue=df_reduce['Clusters'],palette='viridis')
    plt.title("Scatterplot of Agglomerative Clustering of different data points ")
    st.pyplot(fig1)
    st.divider()
    st.code(df_reduce1['Clusters'].value_counts())
    st.divider()
    st.title("Cluster Insights")
    st.header("Cluster 0 – “Quiet Copers” ")
    st.text("Mid-career employees who experience moderate stress levels but tend to keep their struggles to themselves. They rarely reach out for professional or workplace help, partly because the support systems and benefits available to them are inconsistent or unclear. As a result, they often choose to manage their challenges quietly, even when doing so may affect their well-being over time.")
    st.divider()
    st.header('Cluster 1: "Under-Supported Professionals"')
    st.text("This cluster is made up of individuals who have a low risk score but are not being treated for a mental health condition. They tend to work for tech companies with robust benefits, but their care options and wellness programs are limited, which may contribute to a lack of support for their mental health concerns.")
    st.divider()
    st.header('Cluster 2-"Vulnerable Individuals with Limited Support" ')
    st.text("This cluster represents a group of individuals who are more likely to have a family history of mental illness and are more likely to have a mental health consequence as a result of their work. They have limited access to wellness programs, seek help from their employers, but report their mental health is not being treated. They are also less likely to work remotely and feel their employers do not provide care options for them.")
    st.divider()
    st.header('Cluster 3-"The Open Advocates"')
    st.text("This cluster is characterized by individuals who are more likely to have a family history of mental illness and have experienced mental health consequences at work. However, they have a high support score and are open about their struggles. They have access to numerous benefits, and their employers' actions and values align with their own views on mental health, which is a key factor in their overall feeling of support.")
    st.divider()
    st.header('Cluster 4: "Under-Supported Employees"')
    st.text("This cluster is comprised of employees who are not self-employed and have a high likelihood of family history with mental health conditions. They are not being treated for mental health conditions but often experience work interference and mental health consequences because they work for companies that do not offer wellness programs or care options.")
pg=st.navigation([
    st.Page(Home,title='Home'),
    st.Page(EDA,title='EDA'),
    st.Page(Classification,title='Classification'),
    st.Page(Regression,title='Regression'),
    st.Page(Clustering,title='Clustering')
])
pg.run()


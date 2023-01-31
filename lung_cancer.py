# This is a sample Python script.
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import shap
st.set_option('deprecation.showPyplotGlobalUse', False)
level_prediction = np.array(['Low','Medium','High'])
def preprocess(df):
    df.drop("Patient Id", axis=1, inplace=True)

    # cleaning column names
    df.rename(columns=str.lower, inplace=True)
    df.rename(columns={col: col.replace(" ", "_") for col in df.columns}, inplace=True)
    df["level"].replace({'High': 2, 'Medium': 1, 'Low': 0}, inplace=True)
    return df

def svm(X,Y,df):
    #st.header("You are fully well , don't worry ")
    loaded_model=pickle.load(open('D:/ai_project/trained_model.sav','rb'))
    input=df.to_numpy()
    input_reshaped=input.reshape(1,-1)
    prediction=loaded_model.predict(input_reshaped)
    st.write("prediction result using **svm**:")
    if prediction[0]==0:
      st.write("The possiblity of lung cancer is **Low**")
    elif prediction[0]==1:
      st.write("The possiblity of lung cancer is **Medium**")
    else:
      st.write("The possibility of lung cancer is **High**")
    
    model = RandomForestClassifier()
    model.fit(X, Y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st.header('Feature Importance')
    plt.title('Feature importance based on SHAP values')
    shap.summary_plot(shap_values, X)
    st.pyplot(bbox_inches='tight')
    st.write('---')

    plt.title('Feature importance based on SHAP values (Bar)')
    shap.summary_plot(shap_values, X, plot_type="bar")
    st.pyplot(bbox_inches='tight')
    print(df.describe())
    
def logistic_regression(X,Y,df):
    logr=linear_model.LogisticRegression()
    logr.fit(X,Y)
    input=df.to_numpy()
    input_reshaped=input.reshape(1,-1)
    predicted=logr.predict(input_reshaped)
    st.write("prediction result using **Logistic regression**:")
    st.write(level_prediction[predicted])
    if predicted[0]==0:
      st.write("The possiblity of lung cancer is **Low**")
    elif predicted[0]==1:
      st.write("The possiblity of lung cancer is **Medium**")
    else:
      st.write("The possibility of lung cancer is **High**")
    
    model = RandomForestClassifier()
    model.fit(X, Y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st.header('Feature Importance')
    plt.title('Feature importance based on SHAP values')
    shap.summary_plot(shap_values, X)
    st.pyplot(bbox_inches='tight')
    st.write('---')

    plt.title('Feature importance based on SHAP values (Bar)')
    shap.summary_plot(shap_values, X, plot_type="bar")
    st.pyplot(bbox_inches='tight')
    print(df.describe())
    
def random_forest(X,Y,df):
    model = RandomForestClassifier()
    model.fit(X, Y)
    # Apply Model to Make Prediction
    prediction = model.predict(df)

    st.header('Prediction of Outcome')
    st.write(level_prediction[prediction])
    st.write('---')

    # Explaining the model's predictions using SHAP values
    # https://github.com/slundberg/shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st.header('Feature Importance')
    plt.title('Feature importance based on SHAP values')
    shap.summary_plot(shap_values, X)
    st.pyplot(bbox_inches='tight')
    st.write('---')

    plt.title('Feature importance based on SHAP values (Bar)')
    shap.summary_plot(shap_values, X, plot_type="bar")
    st.pyplot(bbox_inches='tight')
    print(df.describe())
    
    
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    st.write("""
    # Lung Cancer Prediction App (AI Project #Team 8)
    This app predicts  **Lung Cancer**! You can give your custom input in sidebar.
    We used *Random Forest Model* to predict **Lung Cancer** from **LOW** to **HIGH**
    """)
    st.write('---')

    # Loads the Boston House Price Dataset

    df = pd.read_csv('D:/dowload project/Lung-Cancer-Prediction-App/cancer patient data sets.csv', index_col='index')
    df = preprocess(df)
    X = df.drop(['level','chest_pain','coughing_of_blood','fatigue','weight_loss','shortness_of_breath','wheezing','swallowing_difficulty','clubbing_of_finger_nails','frequent_cold','dry_cough','snoring'], axis=1)
    Y = df['level']
    st.sidebar.header('Specify Input Parameters')
    st.sidebar.subheader("Give your own inputs to try this model")
    def user_input_features():
        age = st.sidebar.slider('age', float(X.age.min()), float(X.age.max()), float(X.age.mean()))
        gender = st.sidebar.slider('gender', float(X.gender.min()), float(X.gender.max()), float(X.gender.mean()))
        air_pollution = st.sidebar.slider('air_pollution', float(X.air_pollution.min()), float(X.air_pollution.max()), float(X.air_pollution.mean()))
        alcohol_use = st.sidebar.slider('alcohol_use', float(X.alcohol_use.min()), float(X.alcohol_use.max()), float(X.alcohol_use.mean()))
        dust_allergy = st.sidebar.slider('dust_allergy', float(X.dust_allergy.min()), float(X.dust_allergy.max()), float(X.dust_allergy.mean()))
        occupational_hazards = st.sidebar.slider('occupational_hazards', float(X.occupational_hazards.min()), float(X.occupational_hazards.max()), float(X.occupational_hazards.mean()))
        genetic_risk = st.sidebar.slider('genetic_risk', float(X.genetic_risk.min()), float(X.genetic_risk.max()), float(X.genetic_risk.mean()))
        chronic_lung_disease = st.sidebar.slider('chronic_lung_disease', float(X.chronic_lung_disease.min()), float(X.chronic_lung_disease.max()), float(X.chronic_lung_disease.mean()))
        balanced_diet = st.sidebar.slider('balanced_diet', float(X.balanced_diet.min()), float(X.balanced_diet.max()), float(X.balanced_diet.mean()))
        obesity = st.sidebar.slider('obesity', float(X.obesity.min()), float(X.obesity.max()), float(X.obesity.mean()))
        smoking = st.sidebar.slider('smoking', float(X.smoking.min()), float(X.smoking.max()), float(X.smoking.mean()))
        passive_smoker = st.sidebar.slider('passive_smoker', float(X.passive_smoker.min()), float(X.passive_smoker.max()), float(X.passive_smoker.mean()))
        data = {'age': age,
                'gender': gender,
                'air_pollution': air_pollution,
                'alcohol_use': alcohol_use,
                'dust_allergy': dust_allergy,
                'occupational_hazards': occupational_hazards,
                'genetic_risk': genetic_risk,
                'chronic_lung_disease': chronic_lung_disease,
                'balanced_diet': balanced_diet,
                'obesity': obesity,
                'smoking': smoking,
                'passive_smoker': passive_smoker,
                }
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_features()

    # Main Panel

    # Print specified input parameters
    st.header('Specified Input parameters')
    st.write(df)
    st.write('---')
    
    # Build Regression Model
    
    #start here 
    
    
    #option for different algorithm
    st.header("Select specific algorithm")
    option = st.selectbox(
    '',('Random forest', 'Svm','Logistic Regression'))
    if option=="Random forest":
        random_forest(X,Y,df)
    elif option=="Svm":
        svm(X,Y,df)
    elif option=="Logistic Regression":
        logistic_regression(X,Y,df);
    #st.write('You selected:', option)
    
    


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

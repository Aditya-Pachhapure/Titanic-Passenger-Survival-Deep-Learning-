import streamlit as st
import pandas as pd 
from tensorflow.keras.models import load_model
import pickle 

st.title('Passengers Survival Chance in the Titanic Journey')

pclass = st.slider('Enter the Passenger class for the user' , 1 , 3 )
sex = st.selectbox('Enter the Passenger Gender' , ['male' , 'female'])
age = st.slider("Enter the Passenger's Age" , 0 , 114)
sibsp = st.slider("Enter the Passenger's total number of SIBLING(s) and SPOUSE(s)" , 0 , 8)
parch = st.slider("Enter the Passenger's total number of PARENT(s) and Child(ren)" , 0 , 6)
fare = st.number_input('Enter the Fare of Passenger (Max : 640)' , min_value = 0 , max_value = 640)
embarked = st.selectbox('Enter the Passengers station from where they started the journey' , ['Southampton' , 'Chebourg' , 'Queenstown' ])

data = pd.DataFrame([{
    'Pclass' : pclass,
    'Sex' : sex,
    'Age' : age,
    'SibSp' : sibsp,
    'Parch' : parch,
    'Fare' : fare,
    'Embarked' : embarked
}])

model = load_model('model.h5')

with open('label_encoder.pkl' , 'rb') as file:
    label = pickle.load(file)

with open('onehot_encoder' , 'rb') as file:
    onehot = pickle.load(file)

with open('scaler_pkl' , 'rb') as file:
    scale = pickle.load(file)

data['Sex'] = label.transform(data['Sex'])
embarked = onehot.transform(data[['Embarked']])

embarked = pd.DataFrame(embarked , columns=onehot.get_feature_names_out())

data = pd.concat([data.drop(columns='Embarked') , embarked] , axis=1)

data[['Pclass' , 'Age' , 'SibSp' , 'Parch' , 'Fare' ]] = scale.transform(data[['Pclass' , 'Age' , 'SibSp' , 'Parch' , 'Fare' ]])

y = model.predict(data)

y = y[0][0]

def chance(y):
    if y>0.50:
        return 'The Passenger will survive the Journey'
    else:
        return "The Passenger won't survive the Journey"
    
percent = ((y.round(2))*100)

if st.button('Predict the Survival chance'):
    st.write(f"The Probability of surviving : {percent}%")
    st.write(chance(y))
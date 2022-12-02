import streamlit as st
import pandas as pd
import sklearn
import joblib
import xgboost


st.write("""
# House Price Prediction
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    area = st.sidebar.number_input('Area',min_value=30, max_value=1000)
    room = st.sidebar.number_input('Room',min_value=0, max_value=6)
    parking = st.sidebar.selectbox('Parking:',(1 , 0))
    warehouse = st.sidebar.selectbox('Warehouse:', (1 , 0))
    elevator = st.sidebar.selectbox('Elevator:', (1 , 0))
    address = st.sidebar.selectbox('Choose Your Favorite Neighborhood:', ('Shahran', 'Pardis', 'Shahrake Qods', 'Shahrake Gharb',
       'North Program Organization', 'Andisheh', 'West Ferdows Boulevard',
       'Narmak', 'Saadat Abad', 'Zafar', 'Islamshahr', 'Pirouzi',
       'Shahrake Shahid Bagheri', 'Moniriyeh', 'Velenjak', 'Amirieh',
       'Southern Janatabad', 'Salsabil', 'Zargandeh', 'Feiz Garden',
       'Water Organization', 'ShahrAra', 'Gisha', 'Ray', 'Abbasabad',
       'Ostad Moein', 'Farmanieh', 'Parand', 'Punak', 'Qasr-od-Dasht',
       'Aqdasieh', 'Pakdasht', 'Railway', 'Central Janatabad',
       'East Ferdows Boulevard', 'Sattarkhan', 'Baghestan', 'Shahryar',
       'Northern Janatabad', 'Daryan No', 'Southern Program Organization',
       'Rudhen', 'West Pars', 'Afsarieh', 'Marzdaran', 'Dorous',
       'Sadeghieh', 'Jeyhoon', 'Lavizan', 'Shams Abad', 'Fatemi',
       'Keshavarz Boulevard', 'Kahrizak', 'Qarchak',
       'Northren Jamalzadeh', 'Azarbaijan', 'Persian Gulf Martyrs Lake',
       'Beryanak', 'Heshmatieh', 'Elm-o-Sanat', 'Golestan',
       'Shahr-e-Ziba', 'Pasdaran', 'Gheitarieh', 'Kamranieh', 'Gholhak',
       'Heravi', 'Hashemi', 'Dehkade Olampic', 'Damavand', 'Republic',
       'Zaferanieh', 'Qazvin Imamzadeh Hassan', 'Niavaran', 'Valiasr',
       'Qalandari', 'Amir Bahador', 'Ekhtiarieh', 'Ekbatan', 'Absard',
       'Haft Tir', 'Mahallati', 'Ozgol', 'Tajrish', 'Abazar', 'Koohsar',
       'Hekmat', 'Parastar', 'Lavasan', 'Majidieh', 'Southern Chitgar',
       'Karimkhan', 'Si Metri Ji', 'Karoon', 'Northern Chitgar',
       'East Pars', 'Kook', 'Air force', 'Komeil', 'Azadshahr',
       'Amirabad', 'Dezashib', 'Elahieh', 'Mirdamad', 'Razi', 'Jordan',
       'Mahmoudieh', 'Shahedshahr', 'Mehran', 'Nasim Shahr', 'Tenant',
       'Fallah', 'Eskandari', 'Shahrakeh Naft', 'Ajudaniye', 'Tehransar',
       'Nawab', 'Yousef Abad', 'Northern Suhrawardi', 'Hakimiyeh',
       'Nezamabad', 'Garden of Saba', 'Tarasht', 'Araj', 'Vahidieh',
       'Malard', 'Shahrake Azadi', 'Vanak', 'Tehran Now', 'Darabad',
       'Atabak', 'Sabalan', 'Waterfall', 'Ahang', 'Pishva', 'Ghoba',
       'Southern Suhrawardi', 'Abuzar', 'Dolatabad', 'Hor Square',
       'Taslihat', 'Robat Karim', 'Argentina', 'Seyed Khandan',
       'Shahrake Quds', 'Chidz', 'Khavaran', 'Shoosh', 'Vahidiyeh'))
    data = {'Area': area,
            'Room': room,
            'Parking': parking,
            'Warehouse': warehouse,
            'Elevator' : elevator,
            'Address' : address
            }
    features = pd.DataFrame(data, index=["Hi :)"])
    return features

df2 = user_input_features()

st.subheader('User Input parameters :')
st.write(df2)

model = joblib.load('finale2.joblib')
pred = int(model.predict(df2))

st.subheader('Prediction :')
st.markdown(pred)

st.balloons()




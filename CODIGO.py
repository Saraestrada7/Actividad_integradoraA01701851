pip install matplotlib
pip install pandas
pip install pathlib
pip install numpy
pip install plotly
pip install base64


import datetime 
import pickle 
from pathlib import Path
#from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#import streamlit_authenticator as stauth
import time
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import base64
import plotly
#from streamlit_folium import folium_static
#import folium
#from folium.plugins import HeatMap
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
#from statsmodels.tsa.holtwinters import ExponentialSmoothing
#import hydralit_components as hc
import pickle
from pathlib import Path
from PIL import Image
#from streamlit_extras.add_vertical_space import add_vertical_space 
#from streamlit_extras.metric_cards import style_metric_cards 
#from streamlit_extras.colored_header import colored_header
#from streamlit_metrics import metric, metric_row 
#from streamlit_card import card
#from streamlit_option_menu import option_menu
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import r2_score
st.set_page_config(page_title='SFPD Dashboard ', page_icon=':bar_chart:', layout='wide')


#en esta parte del codigo vamos a agregar la imagen de un policia en la parte de arriba seguido de un titulo y dos subheaders
data = pd.read_csv("police.csv")
df = data.fillna(method='bfill')
image_path = "https://petco-my.sharepoint.com/:i:/r/personal/457032_petco_com/Documents/Desktop/policias-estados-unidos.jpg?csf=1&web=1&e=hnGcZg"
st.image(image_path, use_column_width=True)
st.title("ACTIVIDAD INTEGRADORA")
st.subheader('Police Incident Reports from 2018 to 2020 in San Francisco')
st.subheader('SARA ESTRADA A01701851')
st.sidebar.title('SFPD Category Filter')
mapa=pd.DataFrame()
mapa['Date'] = df['Incident Date']
mapa['Day'] = df['Incident Date']
mapa['Police District'] = df['Police District']
mapa['Neighborhood'] = df['Analysis Neighborhood']
mapa['Incident Category'] = df['Incident Category']
mapa['Incident Subcategory'] = df['Incident Subcategory']
mapa['Resolution'] = df['Resolution']
mapa['lat'] = df['Latitude']
mapa['lon'] = df['Longitude']
subset_data2=mapa


police_district_input=st.sidebar.multiselect(
    'Police District',
    mapa.groupby('Police District').count().reset_index()['Police District'].tolist())
if len(police_district_input)>0:
    subset_data2 = mapa[mapa['Police District'].isin(police_district_input)]

subset_data1=subset_data2
neighborhood_input = st.sidebar.multiselect(
    'Neighborhood',
    subset_data2.groupby('Neighborhood').count().reset_index()['Neighborhood'].tolist())
if len(neighborhood_input) >0:
    subset_data1 = subset_data2[subset_data2['Neighborhood'].isin(neighborhood_input)]
    
    
subset_data = subset_data1
incident_input = st.sidebar.multiselect(
    'Incident Category',
    subset_data1.groupby('Incident Category').count().reset_index()['Incident Category'].tolist())
if len(incident_input) >0:
    subset_data = subset_data1[subset_data1['Incident Category'].isin(incident_input)]
subset_data0=subset_data


st.header('Information display')
st.header('DATABASE OF SPECIFIC YEARS (2018-2020)')
subset_data0
st.markdown("Crime location in San Francisco")
st.map(subset_data)
st.subheader("CRIMES X DAY OF THE WEEK")
st.bar_chart(subset_data0['Day'].value_counts(), color = '#4e5bf2')
st.subheader('CRIMES PER DATE')
st.line_chart(subset_data0['Date'].value_counts(), color = '#4e5bf2')
st.subheader('Type of crimes committed')
st.bar_chart(subset_data0['Incident Category'].value_counts(), color = '#4e5bf2')
agree = st.button('Click to see Incident Subcategories')
if agree:
    st.subheader('Subtype of crimes committed')
    st.bar_chart(subset_data0['Incident Subcategory'].value_counts(), color ='#4e5bf2')
st.markdown('Resolution status')
figi, ax1 = plt.subplots()
labels = subset_data0['Resolution'].unique()
ax1.pie(subset_data0['Resolution'].value_counts(), labels = labels, autopct='%1.1f%%', startangle = 20)
st.pyplot(figi)


#GRAFICA DE PASTEL PARA VISUALIZACION DE PORCENTAJES
asalto_data = subset_data0[subset_data0['Incident Category'] == 'Assault']
asalto_por_vecindario = asalto_data['Neighborhood'].value_counts().reset_index()
asalto_por_vecindario.columns = ['Neighborhood', 'Assault Count']
asalto_por_vecindario['Assault Index'] = asalto_por_vecindario['Assault Count'] / len(asalto_data)
fig_pastel = px.pie(asalto_por_vecindario, values='Assault Count', names='Neighborhood', title='Índice de Asaltos por Vecindario')
st.plotly_chart(fig_pastel)



#GRAFICA DE INCIDENTES CON EL PASO DEL TIEMPO
mission_data = subset_data0[subset_data0['Incident Category'] == 'Homicide']
st.subheader('Homicides commited as time passes')
line_chart_data = mission_data.groupby('Date').size()
st.line_chart(line_chart_data)



# grafica nueva de Histograma de Incidentes por Categoría
st.subheader('Histogram of Incidents by Category')
fig_histogram = px.histogram(
    subset_data0,
    x='Incident Category',
    title='Distribution of Incidents by Category',
    labels={'Incident Category': 'Incident Category', 'count': 'Number of Incidents'},
  color_discrete_sequence=['#4e5bf2']  # Color de la barra
)

fig_histogram.update_layout(
    xaxis_title='Incident Category',
    yaxis_title='Number of Incidents',
    font=dict(family="Arial, sans-serif", size=12, color="#7f7f7f"),
    margin=dict(l=0, r=0, b=0, t=30),
)

st.plotly_chart(fig_histogram)

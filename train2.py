import pandas as pd
import numpy as np
from model2 import multi1
from datetime import date
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
#import seaborn as sns
#import plotly.express as px
#import matplotlib.pyplot as plt
#import pycountry
#import plotly.graph_objects as go

model = multi1()
MODEL_NAME = "Multi Layer Perceptron 1"

#Reading the dataset
coronaVirus_df =  pd.read_csv("covid_19_data.csv",index_col='ObservationDate', parse_dates=['ObservationDate'])
#coronaVirus_df.tail()

#replacing null values in Province/State with Country names
coronaVirus_df['Province/State'].fillna(coronaVirus_df['Country/Region'], inplace=True)

coronaVirus_df.drop(['SNo'], axis=1, inplace=True)
coronaVirus_df.head()

#creating new columns for date, month and time which would be helpful for furthur computation
coronaVirus_df['year'] = pd.DatetimeIndex(coronaVirus_df['Last Update']).year
coronaVirus_df['month'] = pd.DatetimeIndex(coronaVirus_df['Last Update']).month
coronaVirus_df['date'] = pd.DatetimeIndex(coronaVirus_df['Last Update']).day
coronaVirus_df['time'] = pd.DatetimeIndex(coronaVirus_df['Last Update']).time

x = np.array(coronaVirus_df.iloc[:,3:6])
y = np.array(coronaVirus_df.iloc[:,6:9])

test_x = np.array(coronaVirus_df.iloc[:150,3:6])
test_y = np.array(coronaVirus_df.iloc[150:,6:9])


model.fit({'input' : x} , {'targets': y} , validation_set=({'input' : test_x} , {'targets': test_y}), n_epoch=15, batch_size=15, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)



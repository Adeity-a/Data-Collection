#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import DataFrame
from pandas import Series
import numpy as np
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use ("bmh")
religiousplacesData = pd.read_csv(r'D:\all_places_of_worship.csv')
religiousplacesData.head


# In[2]:


pip install sweetviz


# In[3]:


religiousplacesData = pd.read_csv(r'D:\all_places_of_worship.csv')
print("Shape of data=>",religiousplacesData.shape)


# In[4]:


religiousplacesData=religiousplacesData[['objectid','id','name','city','state','zip','county','geolinkid','x','y',
                                         snipp'state_id','subtype','members','attendance','loc_type']]
print("Shape of data=>",religiousplacesData.shape)
religiousplacesData.head(16)


# In[5]:


testdf=pd.pivot_table(religiousplacesData,index=['name'],columns='subtype',values=['members','attendance'])
testdf


# In[6]:


religiousplacesData.isnull().sum() #Let’s see if there are any null values present in our dataset


# In[7]:


religiousplacesData.dropna(inplace=True) #There are a few null values in the dataset. So, let’s drop these null values and proceed further
religiousplacesData.isnull().sum()


# In[8]:


import sweetviz as sweetviz
my_report = sweetviz.analyze([religiousplacesData, "religiousplacesData"],target_feat='id')


# In[9]:


my_report.show_html('Report.html') #create a whole report in form of HTML file


# In[10]:


religiousplacesData.columns


# In[11]:


religiousplacesData['subtype'].unique()


# In[12]:


pip install basemap


# In[13]:


import matplotlib.pyplot as plt


# In[14]:


from mpl_toolkits.basemap import Basemap


# In[15]:


# read in data to use for plotted points

#religiousplacesData = pd.read_csv(r'D:\all_places_of_worship.csv')
#religiousplacesData.head

map_data = religiousplacesData[['y','x','subtype']]

colors = {'CHRISTIAN':'red',
          'MUSLIM':'blue', 
          'BUDDHIST':'green',
          'HINDU':'cyan',
          'JUDAIC':'magenta',
          'OTHER' : 'white'
         }


# In[16]:


# Prepare basemap

plt.figure(figsize=(20,15))

m = Basemap(projection="mill"
            ,llcrnrlat=24
            ,urcrnrlat=50
            ,llcrnrlon=-125
            ,urcrnrlon=-67
            ,resolution='c'
            ,epsg=4269)
m.drawlsmask(land_color='white',ocean_color='aqua',lakes=True)
m.drawcoastlines(linewidth=0.5)
m.drawcountries(linewidth=2.0) 
m.drawstates()

parallels = np.arange(0.,81,10.)
meridians = np.arange(0.,360.,10.)

# Pickup coordinates
# m.plot(lon, lat, 'ro', markersize=2 ,alpha=.05)
m.scatter(religiousplacesData['x'], religiousplacesData['y'], c = religiousplacesData['subtype'].apply(lambda x: colors[x]), marker='.', alpha=0.9)

plt.title("US Places of Worship : Location")
plt.show()


# In[17]:


pip install chart_studio


# In[18]:


import chart_studio


# In[31]:


from chart_studio.plotly import plot, iplot
import plotly.graph_objs as go
import plotly.offline as pyo

trace1 = go.Scatter(
                    x = religiousplacesData.state,
                    y = religiousplacesData.attendance, 
                    mode = "markers",
                    name = "CHRISTIAN",
                    marker = dict(color = 'Orange',symbol=1,size=8)) 
trace2 = go.Scatter(
                    x = religiousplacesData.state,
                    y = religiousplacesData.attendance, 
                    mode = "markers",
                    name = "MUSLIM",
                    marker = dict(color = 'Yellow',symbol=2,size=8)) 

trace3 = go.Scatter(
                   x = religiousplacesData.state,
                    y = religiousplacesData.attendance,  
                    mode = "markers",
                    name = "HINDU",
                    marker = dict(color = 'Green',symbol=3,size=8)) 

trace4 = go.Scatter(
                    x = religiousplacesData.state,
                    y = religiousplacesData.attendance, 
                    mode = "markers",
                    name = "BUDDHIST",
                    marker = dict(color = 'Purple',symbol=4,size=8)) 

trace5 = go.Scatter(
                    x = religiousplacesData.state,
                    y = religiousplacesData.attendance, 
                    mode = "markers",
                    name = "JUDAIC",
                    marker = dict(color = 'Blue',symbol=5,size=8)) 
trace6 = go.Scatter(
                    x = religiousplacesData.state,
                    y = religiousplacesData.attendance, 
                    mode = "markers",
                    name = "OTHERS",
                    marker = dict(color = 'Pink',symbol=6,size=8)) 

data = [trace1, trace2, trace3, trace4, trace5, trace6]

layout = dict(title = 'Religion wise attendance of the members', 
              xaxis= dict(title= 'Name',ticklen= 10,zeroline= False,zerolinewidth=1,gridcolor="lightgrey"),
              yaxis= dict(title= 'Members',ticklen= 10,zeroline= False,zerolinewidth=1,gridcolor="lightgrey",),
              paper_bgcolor='lightgrey',
              plot_bgcolor='white')
font=dict(family="DejaVu Sans", size=15,color="black")
fig = dict(data = data, layout = layout)
pyo.offline.iplot(fig, filename='style-scatter')


# In[ ]:





# In[ ]:





# In[ ]:





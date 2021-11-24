from django.http import HttpResponse
from django.shortcuts import render
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime

sns.set()
pal = sns.hls_palette(10, h=.5)
sns.set_palette(pal)

#Avoid display of scientific notation and show precision of 4 decimals:
pd.set_option('display.float_format', lambda x: '%.4f' % x)

#import urllib.request
#url = 'https://s3.amazonaws.com/nyc-tlc/misc/uber_nyc_data.csv'
#filename = 'uber_nyc_data.csv'
#urllib.request.urlretrieve(url, filename)


df_uber = pd.read_csv('uber_nyc_data.csv')
infor = df_uber.info() 
head = df_uber.head()

import datetime

def home(request):
     return render(
        request,
        'hello/Home.html',
        {
            
        }
    )

def dashboard (request):
     return render(
        request,
        'hello/Dashboard.html',
        {
            'head': head
        }
    )
def aboutus(request):
    return render(
        request,
        'hello/Aboutus.html',
        {
            'date': datetime.datetime.now()
        }
    )
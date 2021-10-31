from django.http import HttpResponse
from django.shortcuts import render
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
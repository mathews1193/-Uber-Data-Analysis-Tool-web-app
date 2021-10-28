from django.http import HttpResponse

def home(request):
    return HttpResponse(
         "<h1> Uber Data Analysis! </h1>")

def dashboard (request):
    return HttpResponse(
         "<h1>Dashboard </h1>")

def aboutus(request):
    return HttpResponse(
         "<h1> About US </h1>")
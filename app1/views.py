from django.shortcuts import render
from . import machine_learning
import csv
from django.template.defaulttags import register


# Create your views here.
...
@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)

...
@register.filter
def get_f1_score(dictionary):
    return dictionary.get("f1-score")

def home(request):
    if request.method == 'POST':
        response = dict(machine_learning.preprocessing(request.FILES.get('file')))
        accuracy = response.get('classification_report').get("accuracy")
        return render(request, 'app1/result.html', context={
            "result" : response,
            "accuracy" : accuracy
        })
    return render(request, 'app1/home_page.html')

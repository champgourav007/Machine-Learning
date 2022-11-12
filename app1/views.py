from django.shortcuts import render
from . import machine_learning, algorithm_main
import csv, io
from django.template.defaulttags import register

# Create your views here.
...
@register.filter
def get_item(dictionary:dict, key:str):
    return dictionary.get(key)

...
@register.filter
def get_f1_score(dictionary:dict):
    return dictionary.get('f1-score')

def home(request):
    if request.method == 'POST':
        data = request.POST
        
        header = ['Age','Internships','CGPA','Hostel','HistoryOfBacklogs','Stream_Computer Science','Stream_Electrical','Stream_Electronics And Communication','Stream_Information Technology','Stream_Mechanical','Gender_Male']
        streams_dict = {
           'cs':0,
           'ec':0,
           'it':0,
           'ece':0, 
           'me':0,
        }
        
        streams_dict[data['stream']] = 1
            
        row = [
            data.get('age'), 
            data.get('internships'), 
            data.get('cgpa'), 
            1 if data.get('hostel') == 'on' else 0, 
            1 if data.get('historyOfBacklogs') == 'on' else 0,
            streams_dict['cs'],
            streams_dict['ec'],
            streams_dict['ece'],
            streams_dict['it'],
            streams_dict['me'],
            1 if data.get('gender') == 'Male' else 0
            ]

        with open('form.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(row)
             
        is_placed = algorithm_main.preprocessing()
        print(is_placed)
        return render(request, 'app1/prediction.html', context={
            'result' : is_placed,
            'header' : header,
            'row' : row,
        })
        
    return render(request, 'app1/home_page.html')

import sys
from polls import sumModule
from polls import ChurnModel
#import sumModule

from django.http import HttpResponse
from django.shortcuts import render


def index(request):
    return render(request,'polls/home.html')
    #return HttpResponse("Hello, world. You're at the polls index.")

def search(request):
    #return HttpResponse("Its Working")
    if request.method == 'POST':
        search_id = request.POST.get('textfield', None)
        try:
            #do something with user
            search_id = sumModule.sum(search_id,"5")
            html = "<H1>User Name is " + str(search_id) + " </H1>"
            return HttpResponse(html)
        except:
            return HttpResponse("no such user 1")  
    else:
        return render(request, 'form.html')

def predict(request):
    #return HttpResponse("Its Working")
    if request.method == 'POST':
                
        first_name = request.POST.get('firstName', None)              
        last_name = request.POST.get('lastName', None)
        satisfaction = request.POST.get('satisfaction',None)
        evaluation = request.POST.get('evaluation',None)
        projectcount = request.POST.get('projectCount', None)
        department = request.POST.get('department', None)
        average_montly_hours = request.POST.get('average_montly_hours', None)
        time_spend_company = request.POST.get('time_spend_company', None)
        work_accident = request.POST.get('work_accident', None)
        promotion = request.POST.get('promotion', None)
        salary = request.POST.get('salary', None)

        output = ChurnModel.AttritionPredictor(  satisfaction,
                                        evaluation,
                                        projectcount,
                                        average_montly_hours,
                                        time_spend_company,
                                        work_accident,
                                        promotion,
                                        salary,
                                        department)

        try:
            #do something with user
            #search_id = sumModule.sum(first_name,last_name + " " + str(department))
            #return render(request, 'polls/home.html',output)
            html = "<H1> Employee " + first_name + " " + last_name + " is going to " + output +" </H1>"
            return HttpResponse(html)
        except:
            return HttpResponse("no such user")  
    else:
        return render(request, 'polls/home.html')
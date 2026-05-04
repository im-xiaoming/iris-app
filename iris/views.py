from django.shortcuts import render, redirect
from .forms import IrisForm, DataForm
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
import uuid
import os
from django.conf import settings

# Create your views here.
def home(request):
    if request.method == 'POST':
        form = IrisForm(request.POST)
        if form.is_valid():
            request.session['result'] = 'setosa'
            return redirect('home')
    else:
        form = IrisForm()
    return render(request, 'iris/home.html', {
        'form': form,
        'result': request.session.pop('result', None)
    })
    
    
def upload(request):
    if request.method == 'POST':
        form = DataForm(request.POST, request.FILES)
        if form.is_valid():
            file = form.cleaned_data['data_file']
            upload_path = os.path.join(settings.BASE_DIR, 'data/raw')

            fs = FileSystemStorage(location=upload_path)
            filename = f"{uuid.uuid4()}_{file.name}"
            fs.save(filename, file)
            return HttpResponse('File uploaded successfully')
            
    else:
        form = DataForm()
    return render(request, 'iris/upload.html', {'form': form})
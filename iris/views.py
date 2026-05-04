from django.shortcuts import render, redirect
from .forms import IrisForm

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
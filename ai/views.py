import os
import inspect
from django.shortcuts import render
from django.shortcuts import render, redirect
from .forms import ProfileForm
from .models import Profile
from ai.backend import utils
import ai


# Create your views here.
def index(request):
    pass


def upload_file(request):
    if request.method == 'POST':
        form = ProfileForm(request.POST, files=request.FILES)
        if form.is_valid():
            image = form.save()
            # return redirect('upload_file')
    else:
        form = ProfileForm()
    last_id = Profile.objects.count()
    images = Profile.objects.filter(id=last_id)

    return render(request, 'ai/home.html', {'form': form, 'images': images})


def display_history(request):
    if request.method == 'GET':
        form = ProfileForm(request.POST, files=request.FILES)  # 2
    else:
        form = ProfileForm()
    images = Profile.objects.order_by('-id')
    for image in images:
        image.result = utils.get_result(os.path.join(os.path.dirname(os.path.dirname(inspect.getfile(ai))), 'media', str(image.image)))

    return render(request, 'ai/history.html', {'images': images})

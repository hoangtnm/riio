import os

from django.shortcuts import render

from ai.backend import utils
from .forms import ProfileForm
from .models import Profile


# Create your views here.
def home(request):
    return render(request, "ai/home.html")


def upload(request):
    if request.method == 'POST':
        form = ProfileForm(request.POST, files=request.FILES)
        if form.is_valid():
            image = form.save()
    else:
        form = ProfileForm()
    
    last_id = Profile.objects.count()
    image = Profile.objects.filter(id=last_id)[0]
    
    # Update image'result
    image.result = utils.get_result(
        os.path.join('media', str(image.image)))
    image.save()

    return render(request, 'ai/upload.html', {'form': form, 'image': image})


def history(request):
    if request.method == 'GET':
        form = ProfileForm(request.POST, files=request.FILES)  # 2
    else:
        form = ProfileForm()
    
    images = Profile.objects.order_by('-id')
    return render(request, 'ai/history.html', {'images': images})

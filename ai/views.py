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
            # return redirect('upload_file')
    else:
        form = ProfileForm()
    
    last_id = Profile.objects.count()
    images = Profile.objects.filter(id=last_id)

    return render(request, 'ai/upload.html', {'form': form, 'images': images})


def history(request):
    if request.method == 'GET':
        form = ProfileForm(request.POST, files=request.FILES)  # 2
    else:
        form = ProfileForm()
    
    images = Profile.objects.order_by('-id')

    # Predicts whether each image is cat or dog
    for image in images:
        image.result = utils.get_result(
            os.path.join('media', str(image.image)))

    return render(request, 'ai/history.html', {'images': images})

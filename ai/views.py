import os

from django.shortcuts import render

from ai.backend import utils
from .forms import ProfileForm
from .models import Profile


# Create your views here.
def home(request):
    return render(request, "ai/home.html")


def upload(request):
    image = None

    if request.method == 'POST':
        form = ProfileForm(request.POST, files=request.FILES)
        if form.is_valid():
            image = form.save()
        
        # Update image's result
        last_id = Profile.objects.count()
        image = Profile.objects.filter(id=last_id)[0]
        image.result = utils.get_result(
            os.path.join('media', str(image.image)))
        image.save()

    else:
        form = ProfileForm()

    return render(request, 'ai/upload.html', {'form': form, 'image': image})


def history(request):
    images = Profile.objects.order_by('-id')
    return render(request, 'ai/history.html', {'images': images})

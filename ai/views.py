from django.shortcuts import render
from django.shortcuts import render, redirect
from .forms import ProfileForm
from .models import Profile


# Create your views here.
def home(request):
    return render(request, "ai/home.html")


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

    return render(request, 'ai/upload.html', {'form': form, 'images': images})


def display_history(request):
    if request.method == 'GET':  # 1
        form = ProfileForm(request.POST, files=request.FILES)  # 2
    else:
        form = ProfileForm()
    images = Profile.objects.order_by('-id')
    return render(request, 'ai/history.html', {'images': images})

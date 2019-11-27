from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from .models import Profile
from .forms import ProfileForm
from django.template import loader


# def upload_file(request):
#     if request.method == 'POST':  # 1
#         form = ProfileForm(request.POST, files=request.FILES)  # 2
#         if form.is_valid():  # 3
#             image = form.save()
#             return redirect('upload_file')  # 4
#     else:
#         form = ProfileForm()
#     last_id = Profile.objects.count()
#     images = Profile.objects.filter(id=last_id)
#     return render(request, 'upload/home.html', {'form': form, 'image': images})

def upload_file(request):
    if request.method == 'POST':  # 1
        form = ProfileForm(request.POST, files=request.FILES)  # 2
        if form.is_valid():  # 3
            image = form.save()
            return redirect('upload_file')  # 4
    else:
        form = ProfileForm()

    last_id = Profile.objects.count()
    images = Profile.objects.filter(id=last_id)

    return render(request, 'upload/home.html', {'form': form, 'images': images})


def display_history(request):
    # template = loader.get_template('upload/history.html')
    if request.method == 'GET':  # 1
        form = ProfileForm(request.POST, files=request.FILES)  # 2
    else:
        form = ProfileForm()
    images = Profile.objects.order_by('-id')
    return render(request, 'upload/history.html', {'images': images})

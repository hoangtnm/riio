from django.db import models

# Create your models here.
from django.db import models


class Profile(models.Model):
    name = models.CharField(max_length=100)
    image = models.ImageField(upload_to='images/%Y/%m/%d/', blank=True, null=True)
    result = models.CharField(max_length=50)

    def __str__(self):
        return self.name

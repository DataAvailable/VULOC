from django.db import models

# Create your models here.
from django.db import models


# Create your models here.
class User(models.Model):
    u_name = models.CharField(max_length=20)
    u_password = models.CharField(max_length=255)
    ticket = models.CharField(max_length=20)

    class Meta:
        db_table = 'p427_user'
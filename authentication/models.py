from django.db import models
from django.contrib.auth import get_user_model


User = get_user_model()


class Ear(models.Model):

    # ear id is that unique string which is based on Name, father's name and DOB
    # if this ear matches with another provided ear this string will be returned for reference.
    ear_id = models.CharField(max_length=12, unique=True)

    features_1 = models.TextField(max_length=1000000)
    features_2 = models.TextField(max_length=1000000)
    features_3 = models.TextField(max_length=1000000)
    features_4 = models.TextField(max_length=1000000)

    entries = models.PositiveBigIntegerField(default=0)

    # IMPORTANT
    # user referes to the users of the app - in this case the Site or Vendor.
    # this does not refer to the person whoes ear this is.
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.ear_id}"

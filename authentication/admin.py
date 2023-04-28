from django.contrib import admin
from authentication.models import Ear




@admin.register(Ear)
class EarAdmin(admin.ModelAdmin):
    
    exclude = ('features_1', 'features_2', 'features_3', 'features_4')
from django.conf.urls.static import static
from django.conf.urls import include
from django.contrib import admin
from django.urls import path

from mlflow_employee import settings
from prediction.views import *

urlpatterns = [
    path('about/', about_html, name='about'),
    path('', prediction_html, name='prediction'),
    path('result/', post_result, name='result'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

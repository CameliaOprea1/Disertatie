
from main import views
from django.urls import path, include
from django.conf import settings
from django.contrib import admin
#from django.conf.urls import static

app_name = "main"

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'), # Home page view
    path('', include('route_application.urls')), # Route planner URLs
    
]


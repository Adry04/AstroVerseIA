from django.urls import path
from ia_backend.api import views

urlpatterns = [
    path('get-spaces/', views.GetSpaces.as_view(), name='get_spaces'),
]
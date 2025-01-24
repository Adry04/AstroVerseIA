from django.urls import path
from api import views

urlpatterns = [
    path('get-spaces/<int:pagina>/', views.GetSpaces.as_view(), name='get_spaces'),
]
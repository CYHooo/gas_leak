from django.urls import path
from . import views
urlpatterns = [
    # path('', views.map_detail, name='map_detail'),
    path('', views.contour, name='contour'),
    path('test/', views.test,name='test'),
    path('contour/', views.contour, name='contour')

]
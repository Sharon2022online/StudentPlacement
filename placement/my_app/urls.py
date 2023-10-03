from .import views
from django.urls import path
app_name="my_app"
urlpatterns = [
    path('', views.index, name="index"),
    path('predict_result/', views.predict_result, name='predict_result'),  # Predict result page
]
from django.urls import path
from .views import ItemListCreateView


urlpatterns = [
    path('items/', ItemListCreateView.as_view({'get': 'list', 'post': 'create'}), name='item-list-create'),
    path('items/<int:pk>/', ItemListCreateView.as_view({'get': 'retrieve', 'put': 'update', 'delete': 'destroy'}), name='item-detail'),
]
from rest_framework import serializers
from .models import WebsiteInfo

class WebsiteInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = WebsiteInfo
        fields = '__all__'
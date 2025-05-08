from django.db import models

# Create your models here.

class WebsiteInfo(models.Model): 
    title = models.CharField(max_length=255, blank=True, null=True)
    url = models.URLField(unique=True)
    image = models.ImageField(upload_to='logos/', blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    category = models.CharField(max_length=100, blank=True, null=True)
    tags = models.JSONField(blank=True, null=True)
    ranking = models.IntegerField(default=0)
    rating = models.DecimalField(max_digits=5, decimal_places=4, default=0.0)
    trust_score = models.IntegerField(default=0)
    parameters = models.JSONField(blank=True, null=True)
    sentence_count = models.IntegerField(default=0)
    rating_sum = models.DecimalField(max_digits=10, decimal_places=6, default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


    def __str__(self):
        return self.title
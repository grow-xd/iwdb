from decimal import Decimal
from rest_framework.viewsets import ModelViewSet
from rest_framework.response import Response
from rest_framework import status
from .models import WebsiteInfo
from .serializers import WebsiteInfoSerializer
from urllib.parse import urlparse

def extract_title(url):
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.netloc or parsed_url.path  # handle cases like 'google.com' without scheme
        if hostname.startswith('www.'):
            hostname = hostname[4:]
        title = hostname.split('.')[0].capitalize()
        return title
    except Exception:
        return None

class ItemListCreateView(ModelViewSet):
    queryset = WebsiteInfo.objects.all()
    serializer_class = WebsiteInfoSerializer

    def create(self, request, *args, **kwargs):
        url = request.data.get('url')
        score = request.data.get('score')  # 0 to 1 float
        label = request.data.get('class')  # "good" or "bad"

        print(f"Received URL: {url}, Score: {score}, Class: {label}")

        if not url or score is None:
            return Response({"error": "url and score are required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            score = Decimal(score)
        except:
            return Response({"error": "Score must be a decimal number."}, status=status.HTTP_400_BAD_REQUEST)

        website, created = WebsiteInfo.objects.get_or_create(url=url)

        website.rating_sum += score
        website.sentence_count += 1
        website.rating = website.rating_sum / website.sentence_count        # Extract title from URL
        try:
            title_start = extract_title(url)
            if title_start:
                website.title = title_start
            else:
                return Response({"error": "Invalid URL format."}, status=status.HTTP_400_BAD_REQUEST)
        except (IndexError, AttributeError):
            return Response({"error": "Invalid URL format."}, status=status.HTTP_400_BAD_REQUEST)

        # Optional: update trust_score or ranking based on class label
        if label == 'good':
            website.trust_score += 1
        elif label == 'bad':
            website.trust_score -= 1
        

        website.save() 
        serializer = self.get_serializer(website)
        return Response(serializer.data, status=status.HTTP_200_OK if not created else status.HTTP_201_CREATED)
    
    def list(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        data = []

        for website in queryset:
            
            serialized = self.get_serializer(website).data
            if(website.sentence_count!=0):
                serialized["trust_score"] = round(website.trust_score/website.sentence_count)
                serialized["trust_score"] = website.rating*100
            serialized["rating"] = round(website.rating*5)
            data.append(serialized)

        return Response(data)

from decimal import Decimal
from rest_framework.viewsets import ModelViewSet
from rest_framework.response import Response
from rest_framework import status
from .models import WebsiteInfo
from .serializers import WebsiteInfoSerializer

class ItemListCreateView(ModelViewSet):
    queryset = WebsiteInfo.objects.all()
    serializer_class = WebsiteInfoSerializer

    def create(self, request, *args, **kwargs):
        url = request.data.get('url')
        score = request.data.get('score')  # 0 to 1 float
        label = request.data.get('class')  # "good" or "bad"

        if not url or score is None:
            return Response({"error": "url and score are required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            score = Decimal(score)
        except:
            return Response({"error": "Score must be a decimal number."}, status=status.HTTP_400_BAD_REQUEST)

        website, created = WebsiteInfo.objects.get_or_create(url=url)

        website.rating_sum += score
        website.sentence_count += 1
        website.rating = website.rating_sum / website.sentence_count
        # Extract title from URL
        try:
            title_start = url.split('www.')[1].split('.')[0]
            website.title = title_start.capitalize()
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

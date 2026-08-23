import os
import sys

# Add django_project directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "django_project"))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_project.settings")

from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()

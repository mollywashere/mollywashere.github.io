import os
from threading import Thread
from django.core.wsgi import get_wsgi_application
from django.conf import settings

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'logger.settings')
application = get_wsgi_application()

# Start your logger script in a separate thread
def start_logger():
    import logger_script
    
    logger_script.run_script()

if __name__ == '__main__':
    # Create a separate thread for running the logger script
    script_thread = Thread(target=start_logger)
    script_thread.start()

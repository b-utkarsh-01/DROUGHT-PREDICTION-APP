from flask import Flask

app = Flask(__name__)

from app import routes  # Import routes to handle web and API requests
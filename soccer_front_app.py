import os
import logging
import streamlit as st
from multipage import MultiPage
import eda_app
import prediction_app
import empty_front
# Create Streamlit Application
app = MultiPage()

# Add Logger
LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=LOGLEVEL)
logger = logging.getLogger()

# Title of the main page
st.title('Soccer : what is the final ranking ?')
st.markdown("##### Data comes from l'Équipe website and runs from season 2004-2005 to 2018-2019")

# Add all your applications (pages) here
app.add_page("Menu", empty_front.app)
app.add_page("EDA", eda_app.app)
app.add_page("Prediction", prediction_app.app)

# The main my_app
app.run()

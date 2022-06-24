import os
import logging
import streamlit as st
from multipage import MultiPage
import eda_app as eda_app
import prediction_app as prediction_app
import eda_new_app as eda_new_app
import empty_front as empty_front

# Create Streamlit Application
app = MultiPage()

# Add Logger
LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=LOGLEVEL)
logger = logging.getLogger()

# Title of the main page
st.title('Soccer : what is the final ranking ?')
st.markdown(" Data comes from l'Équipe website and goes from season 2004-2005 to 2018-2019")

# Add all your applications (pages) here
app.add_page("Menu", empty_front.app)
app.add_page("EDA", eda_app.app)
app.add_page("EDA with ongoing season", eda_new_app.app)
app.add_page("Prediction", prediction_app.app)

# The main my_app
app.run()

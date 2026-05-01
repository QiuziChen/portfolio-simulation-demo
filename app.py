"""Streamlit app entry point.

Run locally:
    streamlit run app.py

Deploy:
    Push to GitHub and connect the repo on Streamlit Community Cloud
    (https://share.streamlit.io), setting the main file path to ``app.py``.
"""
from sim.dashboard import _run_app

_run_app()

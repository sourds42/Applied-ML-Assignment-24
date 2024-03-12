import score
import numpy
import os
import requests
import subprocess
import time
import unittest
import joblib
import pytest

# threshold=0.7

# label,prop=score.score(sent,model,threshold)

@pytest.fixture

def model():
        
    model_path = r"best_model.joblib"
    trained_model=joblib.load(model_path)

    # sent="Congratulations! You have won a free ticket to the cinema!"
    return trained_model

def test_smoke(model):
    threshold=0.5
    sent="Congratulations! You have won a free ticket to the cinema!"

    label,prop=score.score(sent,model,threshold)

        
    assert label in [0,1]
    assert 0 <= prop <=1

def test_format(model):
    threshold=0.5
    sent="Congratulations! You have won a free ticket to the cinema!"

    label,prop=score.score(sent,model,threshold)

    assert isinstance(label, int)
    assert isinstance(prop, float)


def test_threshold_0(model):
    threshold=0
    sent="Congratulations! You have won a free ticket to the cinema!"
    label,prop=score.score(sent,model,threshold)
    assert label==1


def test_threshold_1(model):
    threshold=1
    sent="Congratulations! You have won a free ticket to the cinema!"
    label,prop=score.score(sent,model,threshold)
    assert label==0

def test_spam(model):
    threshold=0.5
    sent="Congratulations! You have won a free ticket to the cinema!"
    label,prop=score.score("YOU HAVE WON 1 MILLION DOLLARS. SEND YOUR ACCOUNT DETAILS!",model,threshold)
    assert label == 1


def test_ham(model):
    threshold=1
    sent="Its a real mail, not spam"
    label,prop=score.score("Dogs are better than cats anyday.",model,threshold)
    assert label == 0

def test_flask():
    # Launch the Flask app using os.system
    os.system('start /b python app_new.py')

    # Wait for the app to start up
    time.sleep(15)

    # Make a request to the endpoint
    response = requests.get('http://127.0.0.1:5000/')
    print(response.status_code)

    # # Assert that the response is what we expect
    # self.assertEqual(response.status_code, 200)
    # print("OK Checked!")
    # self.assertEqual(type(response.text), str)
    # print("OKAY Final Check Done!")

    assert response.status_code == 200
    assert type(response.text)== str

    # Shut down the Flask app using os.system
    os.system('kill $(lsof -t -i:5000)')


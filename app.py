import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input
import os
from keras.models import load_model
import requests
from bs4 import BeautifulSoup

# Load the model and labels
model = load_model('FV.h5')
# Load the labels
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}
# Load the fruits and vegetables
fruits = ['Apple', 'Banana', 'Bell Pepper', 'Chilli Pepper', 'Grapes', 'Jalapeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Radish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']

# Function to fetch calories from Google search
def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        return "Unable to fetch the Calories"

# Function to predict the condition
def predict_condition(prediction, model_name):
    try:
        img = load_img(prediction, target_size=(224, 224))  # Load the image
        img_array = img_to_array(img)  # Convert image to array
        img_array = preprocess_input(img_array, model_name)  # Preprocess the image (based on the model used)
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match the input shape of the model

        prediction_probabilities = model.predict(img_array)[0]  # Make prediction
        result_to_show = "*Depending on keep, the Shelf life can be extended between :(" + str(
            round(np.min(prediction_probabilities) * 24, 4)) + " - " + str(
            round(np.max(prediction_probabilities) * 24, 4)) +" days)" 
        Mean_days_rem = "Mean Days Remaining : " + str(
            round(np.mean(prediction_probabilities) * 24, 2))

        return result_to_show, Mean_days_rem
    except Exception as e:
        return "Unable to predict the condition"


# Function to preprocess the image
def preprocess_input(img_array, model_name):
    if model_name == 'resnet50':
        return resnet_preprocess_input(img_array)
    elif model_name == 'densenet':
        return densenet_preprocess_input(img_array)
    elif model_name == 'inception_v3':
        return inception_preprocess_input(img_array)
    elif model_name == 'mobilenet':
        return mobilenet_preprocess_input(img_array)
    # Add more cases for other models as needed
    else:
        raise ValueError("Invalid model name. Supported models: 'resnet50', 'densenet', 'inception_v3', 'mobilenet', etc.")

    

# Function to prepare the image for prediction
def prepare_image(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = int(y_class)
    res = labels[y]
    return res.capitalize()

# Define the app and layout
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Fruits🍍-Vegetable🍅 Classification"),
    dcc.Input(id='image-path-input', type='text', placeholder='Enter image path...', style={"width": "100%", "padding": "12px 20px", "margin": "8px 0", "box-sizing": "border-box", "border": "3px solid #555"}),
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'ResNet50', 'value': 'resnet50'},
            {'label': 'DenseNet', 'value': 'densenet'},
            {'label': 'Inception V3', 'value': 'inception_v3'},
            {'label': 'MobileNet', 'value': 'mobilenet'}
        ],
        value='densenet',  # Default value
        style={"width": "100%", "padding": "12px 20px", "margin": "8px 0", "box-sizing": "border-box", "border": "3px solid #555"}
    ),
    html.Button('Submit', id='submit-button', style={"width": "100%", "fontSize": "20px", "backgroundColor": "#04AA6D", "border": "none", "color": "white", "padding": "16px 32px", "textDecoration": "none", "margin": "4px 2px", "cursor": "pointer"}),
    # Component for displaying the uploaded image
    html.Div(id='output-image-upload'),
    # Component for displaying the prediction result
    html.Div(id='prediction-output'),
    # Component for displaying the category (fruit or vegetable)
    html.Div(id='category-output'),
    # Component for displaying the condition (good or bad)
    html.Div(id='condition-output'),
    # Component for displaying the calories
    html.Div(id='calories-output')
])

# Callback to process the provided image path and display the results
@app.callback(
    [Output('output-image-upload', 'children'),
     Output('prediction-output', 'children'),
     Output('category-output', 'children'),
     Output('condition-output', 'children'),
     Output('calories-output', 'children')],
    [Input('submit-button', 'n_clicks')],
    [State('image-path-input', 'value'),
    State('model-dropdown', 'value')]
)
def update_output(n_clicks, image_path, selected_model):
    if n_clicks and image_path:  
        if os.path.exists(image_path):  
            img = Image.open(image_path)
            img = img.resize((250, 250))

            prediction = prepare_image(image_path)
            is_fruit = prediction in fruits

            condition = predict_condition(image_path, selected_model)
            
            
            fig = px.imshow(np.array(img))
            fig.update_layout(margin=dict(l=10, r=10, t=0, b=0))

            return html.Div([
                html.H5("Uploaded Image"),
                dcc.Graph(figure=fig)
            ]), html.Div([
                html.H1("Prediction"),
                html.H3(prediction)
            ]), html.Div([
                html.H1("Category"),
                html.H3("Fruit" if is_fruit else "Vegetable")
            ]), html.Div([
                html.H1("Condition, As per ML Models"),
                html.H3(condition[0]),
                html.P(""),
                html.H3(condition[1])
                
            ]), html.Div([
                html.H1("Calories, As per google.com scraper"),
                html.H3(fetch_calories(prediction))
            ])
        else:
            return html.Div("Image path does not exist."), dash.no_update, dash.no_update, dash.no_update, dash.no_update

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)

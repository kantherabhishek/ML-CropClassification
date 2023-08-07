# ML-CropClassification
# Crop Classification with Machine Learning

![Crop Classification]([https://your-image-link.com](https://github.com/kantherabhishek/ML-CropClassification/image.jpg))

## Overview

This repository contains a Python code example that demonstrates how machine learning can be applied to classify different crops, specifically fruits and vegetables. The code utilizes a pre-trained machine learning model to predict whether an uploaded image contains a fruit or a vegetable. Additionally, it provides insights into the condition of the produce and estimates its shelf life.

## Features

- Automated classification of crops based on uploaded images.
- Prediction of the category (fruit or vegetable) for better sorting and pricing.
- Estimation of the condition and remaining shelf life of the produce.
- Fetching of calorie information from Google search for the predicted crop.
- User-friendly web interface for easy interaction.

## How to Use

1. Clone this repository to your local machine using `git clone https://github.com/kantherabhishek/ML-CropClassification.git`.

2. Install the required Python packages using `pip install`.

3. Run the application using `python app.py`.
  ### Requirements

- Python 3.x
- Dash
- Plotly
- Pillow
- TensorFlow
- Keras
- NumPy
- requests
- BeautifulSoup
4. Access the web application by opening a web browser and navigating to `http://localhost:8050`.

5. Enter the path of an image containing a crop and select a model from the dropdown menu (ResNet50, DenseNet, Inception V3, or MobileNet).

6. Click the "Submit" button to see the results, including the uploaded image, crop prediction, category, condition, and calorie information.



## Customization

- You can customize the list of fruits and vegetables by modifying the `fruits` and `vegetables` lists in the code.
- The `labels` dictionary maps class indices to fruit and vegetable names and can also be modified if needed.
- Additional preprocessing functions can be added for other machine learning models.

## Contributing

Contributions to this project are welcome! If you have any suggestions, improvements, or bug fixes, feel free to open an issue or create a pull request.

## License

This project is licensed under the MIT License. Feel free to use and modify the code as needed. Refer to the [LICENSE](LICENSE) file for more details.


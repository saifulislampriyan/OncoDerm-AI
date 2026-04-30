# skin-cancer-detection
this repo contains code for skin cancer detection.

## Installation
Create a virtual environment from pycharm editor. After activating the virtual environment run the following command.
```commandline
pip install -r requirements.txt
```

## Train the model
Run model_train.py to train the model. 

Change batch_size in line 29, and epochs in line 60 to resolve overfitting or underfitting.

Comment out either one of these line to save the model in .h5 or .keras model
```python
# Save the trained model
#model.save('cancer_detection_model.h5')  ## run this line if saving model in .keras is giving any error
#model.save('cancer_detection_model.keras')  ## run this to save model in .keras format
```

## Evaluate the model
Run model_eval.py to evaluate the model on test dataset.

Based on saved model format, either comment out line with .h5 or .keras model.
```python
# Load the trained model
#model = load_model('cancer_detection_model.h5')  ## use this if model is saved in .h5
#model = load_model('cancer_detection_model.keras')  ## use this if model is saved in .keras format
```

## Run the app
Run the streamlit app to upload an image to classify cancer.

To do so, first uncomment either one of these lines to load the saved model.
```python
# Load the trained model
#model = load_model('cancer_detection_model.h5')  ## use this if model is saved in .h5
#model = load_model('cancer_detection_model.keras')  ## use this if model is saved in .keras format
```

Then run this command. Web app will pop up on browser.
```commandline
streamlit run app.py
```
# Skin-Care-Recommendation-System
# HARDWARE REQUIREMENTS
1. Windows 10/11    
2. GPU-Graphical Processing Unit     
3. Browser    
# SOFTWARE REQUIREMENTS
1. PyCharm    
2. Jupyter Notebook   
3. Libraries   
1. Numpy   
2. Pandas   
3. Tensorflow   
4. Sklearn   
5. Plotly   
6. Seaborn   
7. Matplotlib   
8. Cv2    
9. Roboflow    
10. Ultralytics


# SKIN TYPE CLASSIFICATION
It is a Binary Classifier to classify the skin types as Oily Skin and Dry skin. A Convolutional
Neural Network is built for classifying the picture entered by the user into the types of skin.
This helps us get the class percentages to give the classified result. This model helps us to
understand the skin type for later improvising the recommendations filtering of the skin
products for the skin type

# ACNE DETECTION AND SEVERITY CLASSIFICATION
The Yolo (You only look once) v8 computer vision model from the ultralytics package is used
to implement acne detection. 

Roboflow is a tool designed to simplify computer vision tasks in deep learning. It provides a platform for labeling images, building models for categorization, and detecting various classes within an image. The program downloads the trained model weights to make predictions on new images provided by the user. Upon testing the trained model, a dictionary with the bounding box coordinates' values will be given. This allows for classification and the visualization of images with the highlighted box coordinates.

# Recommendaton System

The product-based recommendation system is designed to select the most similar products to a chosen item. This analysis aids in creating a website that offers products tailored to different users based on their skin type, concerns, and desired effects. The system establishes a complex relationship between product features using TF-IDF cosine similarity values to match products. TF-IDF (Term Frequency Inverse Document Frequency) calculates the relevance of a word in a series to a given text by determining its frequency. This frequency is represented as a vector. The cosine similarity algorithm then measures the similarity between two products by comparing their TF-IDF vector values. It calculates the cosine angle between these vectors to assess the similarity of the text values. Products with smaller cosine angles are considered more similar. This approach helps in building an effective recommender system.

# UI
After exporting the dataset, the recommendation website is developed using the Streamlit Python library. The similarity values are utilized to recommend products by filtering them based on the user's skin type, skin concerns, and desired effects. The recommender system is constructed similarly to the recommendation model. The web-based application provides the top five product recommendations that best address the user's specific concerns.

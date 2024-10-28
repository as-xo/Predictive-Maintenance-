This project uses a predictive maintenance model to identify possible machinery failures based on simulated sensor data. 

The goal is to predict equipment breakdowns by recognizing patterns in measurements like vibration levels, temperature, and motor speed. This approach can help plan maintenance before issues occur, minimizing downtime.

Key measurements include:
Vibration levels, Temperature, Motor speed


The aim is to help maintenance teams detect problems early, preventing unexpected equipment issues.


Since no real sensor data was available, I simulated some realistic data:

 •	Total Data Points: 1000 observations over a timeline.
 
 •	Features: vibration_level, temperature, and motor_speed.
 
 •	Failure Trend: Gradual increase in vibration levels to indicate an approaching failure.


Algorithm
I chose a Random Forest Classifier for this project because it capture non-linear relationships within data.

Preprocessing

 •	Standardization: Scaled the feature values to make the model more accurate.
 
 •	Train-Test Split: Divided data into training (80%) and testing (20%) sets to evaluate model performance.

 

Training
Features for predictions:

 •	Vibration level
 
 •	Temperature
 
 •	Motor speed
 
Key Results

  •	Accuracy: High accuracy in identifying maintenance needs.
  
  •	Feature Importance: Vibration level and temperature turned out to be the most important factors in predicting failures.


  
Visualizations
1.	Confusion Matrix: Shows how well the predictions matched actual outcomes.
2.	Feature Importance Plot: Highlights which measurements most influenced the predictions.
3.	Predicted vs Actual Failures: Compares model predictions with actual data.

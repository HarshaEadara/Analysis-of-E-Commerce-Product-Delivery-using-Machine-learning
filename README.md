# Analysis of E-Commerce Product Delivery using Machine Learning

This repository contains an analysis and predictive modeling project focused on e-commerce product delivery using machine learning techniques. The project aims to understand the factors affecting product delivery times and build predictive models to enhance delivery performance.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Data Preprocessing](#data-preprocessing)
- [Tools and Technologies](#tools-and-technologies)
- [Workflow](#workflow)
- [Key Insights](#key-insights)
- [Results and Evaluation](#results-and-evaluation)
- [Usage](#usage)
- [Contributing](#contributing)

## Project Overview

Timely product delivery is a critical factor in customer satisfaction for e-commerce platforms. This project uses machine learning models to predict delivery times based on various factors, such as product type, shipment mode, and distance. The analysis provides actionable insights to improve operational efficiency and reduce delays.

## Dataset

The dataset used in this project collected from Kaggle and available to use in `data` folder includes:

- **Product details**: Information about the product category and dimensions.
- **Delivery attributes**: Shipping mode, distance to the customer, and delivery time.
- **Customer information**: Relevant details impacting delivery, such as location.

> **Note**: The dataset has been preprocessed to remove missing values and outliers for effective analysis.

## Features

The analysis includes the following features:

- **Product Category**: Type of product being shipped.
- **Shipping Mode**: Mode of shipment such as air, ground, or sea.
- **Distance**: The distance between the warehouse and customer location.
- **Delivery Time**: Target variable representing the actual delivery time.

## Data Preprocessing

The preprocessing steps applied to the dataset include:

1. Removal of duplicate entries.
2. One-hot encoding of categorical variables.
3. Normalizing numerical features.
4. Preparing the data for model training and evaluation.

## Tools and Technologies

The project leverages the following technologies:

- **Python Libraries**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- **Machine Learning Models**:
  - Random Forest
  - Decision Tree
  - Logistic Regression
  - K Nearest Neighbors (KNN)
- **Visualization**: Data visualizations for exploratory data analysis (EDA).
- **Jupyter Notebook**: Development and analysis environment.

## Workflow

1. **Data Preprocessing**: Cleaning and transforming the dataset to prepare for analysis.
2. **Exploratory Data Analysis (EDA)**: Visualizing relationships between features and identifying trends.
3. **Feature Engineering**: Creating derived features to improve model performance.
4. **Model Development**:
   - Splitting data into training and testing sets.
   - Training and evaluating multiple machine learning models.
5. **Model Evaluation**: Comparing performance metrics such as precision, recall, F1-score, and accuracy.
6. **Deployment-ready Code**: Exporting final model for future deployment.

## Key Insights

- **Product Characteristics**:
  - Products weighing between 2500 and 3500 grams and costing less than $250 are more likely to be delivered on time.
  - Products with a discount of more than 10% are more likely to be delivered on time, while those with a discount of 0-10% are more likely to be delayed.

- **Customer Behavior**:
  - Customers who make frequent calls are more likely to experience delayed deliveries.
  - Returning customers, who have completed more prior transactions, tend to have higher rates of on-time deliveries.

- **Operational Insights**:
  - The majority of shipments originate from Warehouse F, likely located near a seaport, which influences delivery times.

## Results and Evaluation
### Performance of Models
The performance of the models was evaluated based on various metrics including precision, recall, F1-score, and accuracy:
- **Random Forest:** Achieved a precision of 0.72, recall of 0.71, F1-score of 0.68, and an accuracy of 68%.
- **Decision Tree:** Performed the best overall, with a precision of 0.76, recall of 0.73, F1-score of 0.68, and the highest accuracy at 69%.
- **Logistic Regression:** Had a precision of 0.62, recall of 0.62, F1-score of 0.62, and an accuracy of 63%.
- **K Nearest Neighbors:** Obtained a precision of 0.66, recall of 0.65, F1-score of 0.65, and an accuracy of 65%.

### Best Model
Based on the evaluation metrics, the Decision Tree classifier emerged as the best-performing model. With the highest accuracy of 69% and strong precision and recall scores, it provided the most reliable performance among the models tested. The Decision Tree's ability to handle complex decision-making processes and its interpretability make it an excellent choice for this context. The Random Forest classifier also performed well, closely following the Decision Tree in terms of accuracy, making it a strong contender for predictive tasks.

### Conclusion

The project's goal was to forecast whether or not an e-commerce company's merchandise would arrive on time. Based on the analysis, the following conclusions were drawn:

1. Product characteristics, such as weight and price, significantly affect delivery times.
2. Customer behavior plays a role in predicting delivery outcomes, with returning customers showing better delivery rates.
3. Among the machine learning models, the Decision Tree classifier outperformed others in terms of accuracy.
4. The insights derived can assist e-commerce companies in optimizing delivery processes and improving customer satisfaction.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/HarshaEadara/Analysis-of-E-Commerce-Product-Delivery-using-Machine-learning.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Analysis-of-E-Commerce-Product-Delivery-using-Machine-learning
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Analysis_of_E_Commerce_Product_Delivery_using_Machine_learning.ipynb
   ```
5. Ensure the dataset `E_Commerce.csv` is available in the project directory.
6. Run the notebook cells sequentially to reproduce the analysis.

## Contributing

Contributions to this project are welcome. If you'd like to suggest improvements or report issues, please open an issue or submit a pull request on the repository. Let's collaborate to make this project even better!

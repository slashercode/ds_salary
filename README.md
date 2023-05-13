# Predicting Salaries for Data-related Positions

## Project Overview

This repository contains information about a personal data science project aimed at predicting salaries for data-related positions using a **Data Science Salary** dataset from Kaggle for the year 2023.

The dataset contained information on variables such as job title, company location, employee residence, work experience, employment type, remote work ratio, and company size. The goal of the project was to build a regression model that could accurately predict salary in USD for data-related jobs based on these variables.

## Requirements

The following packages are required for this project:

- Pandas
- NumPy
- Seaborn
- Scikit-Learn
- Plotly

It is recommended to install these packages using a package manager such as pip or conda. For example, to install these packages using pip, open a terminal or command prompt and run the following command:

    pip install pandas numpy seaborn scikit-learn plotly

**Note:** Some of these packages may have dependencies that need to be installed as well.

## Project Workflow

The project involved several steps, including data cleaning, exploratory data analysis (EDA), data preprocessing, and model building. I used Python and several libraries, including **Pandas**, **NumPy**, **Seaborn**, **Scikit-Learn**, **Plotly**, to carry out the various tasks. I also implemented four models, namely **Random Forest Regression**, **Decision Tree Regression**, **Gradient Boosting Regression**, and **Linear Regression**, to predict the salaries.

To preprocess the data, the I handled categorical variables such as job title and company location by filtering the dataset to only include data engineer, data scientist, data analyst, and ML engineer jobs and locations within the USA. I also used imputation methods such as **Simple Imputer** to handle missing numerical data and **One-Hot Encoding** to convert categorical variables into numerical form.

Next, I built four regression models using the **Scikit-Learn** library. I used **GridSearchCV** to perform hyperparameter tuning and improve the model performance.

## Model Performance

The table generated at the end shows the performance metrics for four different regression models used in the project.

Based on the table, we can see that the **_Linear Regression_** model performed the **_best_**, with the **_lowest Mean Absolute Error (MAE)_** and **_highest R-squared score_**, both before and after hyperparameter tuning. The Gradient Boosting Regressor also performed relatively well, with the second-lowest MAE and R-squared score. The Random Forest Regression and Decision Tree Regressor models had higher MAE values and lower R-squared scores, indicating that they did not fit the data as well as the other two models.

## Conclusion

There could be several reasons for the outcome observed in the performance metrics. One possible reason is that Linear Regression is a simpler model than the other three models, which could make it more robust and less prone to overfitting. Another reason could be that the features in the dataset have a linear relationship with the target variable (salary), which makes Linear Regression a good fit for the data. On the other hand, the other models may have more complex relationships between the features and target variable, which could cause them to overfit the data and perform poorly on new data. Additionally, the hyperparameter tuning process may have been more effective in finding the optimal parameters for the Linear Regression and Gradient Boosting Regressor models, leading to better performance

## Acknowledgement

You can view the dataset here: [Data Science Salaries 2023](https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023)

## License

**NOT FOR COMMERCIAL USE**

_If you intend to use any of my code for commercial use please contact me and get my permission._

_If you intend to make money using any of my code please get my permission._

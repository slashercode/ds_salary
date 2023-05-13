######################
# IMPORT LIBRARIES
######################
import warnings

import geopandas as gpd
import kaleido
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pycountry
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from wordcloud import STOPWORDS, WordCloud

warnings.filterwarnings("ignore")

# load dataset
data = pd.read_csv("ds_salaries.csv")

# overview of the data
data.head()
print(
    "There are {} observations and {} features in this dataset. \n".format(
        data.shape[0], data.shape[1]
    )
)

columns = data.columns
data[columns].nunique()


######################
# CLEAN THE DATASET
######################

# check for missing values
data.isnull().sum()
# check for duplicates
print(
    "There are in total {} duplicates in this dataset. \n".format(
        data.duplicated().sum()
    )
)
# keep the first duplicate value and drop the rest
data = data.drop_duplicates(keep="first")
#
print(
    "After data cleaning there are {} observations and {} features in this dataset. \n".format(
        data.shape[0], data.shape[1]
    )
)

######################
# VISUALIZATION
######################

# set the background color
bg = "#D5DBDB"
# set the filled color
fill_col = "#D35400"

# modify to only show top 10 jobs
top10_jobs = data.job_title.value_counts().iloc[:10]

# Work Year, Experience Level, Employment Type, Job Title, Remote Ratio & Company Size
# create the 3x2 grid
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 8))

# set the figure properties for each subplot
sns.countplot(x=data["work_year"], color=fill_col, ax=axs[0, 0])
axs[0, 0].set_title("Work Year", size=15, weight="bold")
axs[0, 0].set_xlabel("Year")
axs[0, 0].set_ylabel("Salary")

sns.countplot(x=data["experience_level"], color=fill_col, ax=axs[0, 1])
axs[0, 1].set_title("Experience Level", size=15, weight="bold")
axs[0, 1].set_xlabel("Level of Experience")
axs[0, 1].set_ylabel("Number of Employees")

sns.countplot(x=data["employment_type"], color=fill_col, ax=axs[1, 0])
axs[1, 0].set_title("Employment Type", size=15, weight="bold")
axs[1, 0].set_xlabel("Type of Employment")
axs[1, 0].set_ylabel("Number of Employees")

sns.barplot(
    x=top10_jobs.values, y=top10_jobs.index, color=fill_col, orient="h", ax=axs[1, 1]
)
axs[1, 1].set_title("Top 10 Most Common Job", size=15, weight="bold")
axs[1, 1].set_xlabel("Number of Emplpoyees")
axs[1, 1].set_ylabel("Job Title")

sns.countplot(x=data["remote_ratio"], color=fill_col, ax=axs[2, 0])
axs[2, 0].set_title("Remote Ratio", size=15, weight="bold")
axs[2, 0].set_xlabel("Ratio")
axs[2, 0].set_ylabel("Number of Employees")

sns.countplot(x=data["company_size"], color=fill_col, ax=axs[2, 1])
axs[2, 1].set_title("Company Size", size=15, weight="bold")
axs[2, 1].set_xlabel("Category of Size")
axs[2, 1].set_ylabel("Number of Employees")

# adjust the spacing between subplots
plt.tight_layout()
# save the plot as png image
plt.savefig("fig_1.png")

# create a wordcloud
job_titles = (
    data.groupby("job_title")
    .size()
    .reset_index(name="count")
    .sort_values(by="count", ascending=False)
)
job_titles = pd.DataFrame(job_titles)

stopwords = set(STOPWORDS)


def show_wordcloud(data):
    wordcloud = WordCloud(
        background_color=bg,
        stopwords=stopwords,
        max_words=300,
        max_font_size=30,
        scale=3,
        random_state=1,
        contour_width=5,
        contour_color="steelblue",
        colormap="Blues",
    ).generate_from_frequencies(data.set_index("job_title")["count"].to_dict())

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.set_facecolor(bg)
    plt.axis("off")

    plt.imshow(wordcloud, interpolation="bilinear")
    # save the plot as png image
    plt.savefig("fig_2.png")


show_wordcloud(job_titles)

# Salary
# create the 3x1 grid
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))

# set the figure properties for each subplot
sns.countplot(
    x=data["salary_currency"],
    order=data["salary_currency"].value_counts().index,
    color=fill_col,
    ax=axs[0],
)
axs[0].set_title("Salary Currency by Count", size=15, weight="bold")
axs[0].set_xlabel("Currency Type")
axs[0].set_ylabel("Number of Employees")

sns.histplot(x=data["salary"], color=fill_col, ax=axs[1])
axs[1].set_title("Salary Distribution", size=15, weight="bold")
axs[1].set_xlabel("Salary")
axs[1].set_ylabel("Number of Employees")

sns.histplot(x=data["salary_in_usd"], kde=True, color=fill_col, ax=axs[2])
axs[2].set_title("Salary Distribution in USD", size=15, weight="bold")
axs[2].set_xlabel("Salary in USD")
axs[2].set_ylabel("Number of Employees")


# adjust the spacing between subplots
plt.tight_layout()
# save the plot as png image
plt.savefig("fig_3.png")

# Employee Residence
# convert the employee residence to ISO-3 country code from corresponding alpha-2 country code
iso_3_emp = []
for code in data["employee_residence"]:
    iso_3 = pycountry.countries.get(alpha_2=code).alpha_3
    iso_3_emp.append(iso_3)

# create a new column "ISO-3_emp" in the data DataFrame
# and set its values to the retrieved ISO-3 country codes
data["ISO-3_emp"] = iso_3_emp

# group the data
employee_residence = data.groupby("ISO-3_emp").size().reset_index(name="count")

# load the world map data
world_map = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
# create a choropleth map based on the number of employees in each country
fig = go.Figure(
    data=go.Choropleth(
        locations=employee_residence["ISO-3_emp"],
        z=employee_residence["count"],
        colorscale="reds",
        locationmode="ISO-3",
        marker_line_color="black",
        marker_line_width=0.5,
        colorbar_title="Number of Employees",
    )
)
# set the layout of the map
fig.update_layout(
    title="Employee Residence by Country", width=800, height=600, plot_bgcolor=bg
)
# save the map as a png image
fig.write_image("employee_residence.png", engine="kaleido", format="png")

# Company Location
# convert the company location to ISO-3 country code from corresponding alpha-2 country code
iso_3_comp = []
for code in data["company_location"]:
    iso_3 = pycountry.countries.get(alpha_2=code).alpha_3
    iso_3_comp.append(iso_3)

# create a new column "ISO-3_comp" in the data DataFrame
# and set its values to the retrieved ISO-3 country codes
data["ISO-3_comp"] = iso_3_comp

# group the data
company_location = (
    data.groupby("ISO-3_comp")
    .size()
    .reset_index(name="count")
    .sort_values(by="count", ascending=False)
)
# create a choropleth map based on the number of employees in each country
fig = go.Figure(
    data=go.Choropleth(
        locations=company_location["ISO-3_comp"],
        z=company_location["count"],
        colorscale="reds",
        locationmode="ISO-3",
        marker_line_color="black",
        marker_line_width=0.5,
        colorbar_title="Number of Companies",
    )
)
# set the layout of the map
fig.update_layout(
    title="Company Location by Count", width=800, height=600, plot_bgcolor=bg
)
# save the map as a png image
fig.write_image("company_location.png", engine="kaleido", format="png")


######################
# BUILD THE MODEL
######################
# make a copy of the dataset
data_copy = data.copy()


# handle the categorical data
# Job Title (consider only data engineer, data scientist, data analyst and ml engineer)
def map_job_title(job_title):
    if job_title in [
        "Data Engineer",
        "Data Scientist",
        "Data Analyst",
        "Machine Learning Engineer",
    ]:
        return job_title
    else:
        return "Other"


data_copy["modified_job_title"] = data_copy["job_title"].apply(map_job_title)

# Company Location (consider only USA)
data_copy["company_location"] = np.where(data["company_location"] == "US", 1, 0)
data_copy.rename(columns={"company_location": "company_location_US"}, inplace=True)

# Employee Residence (consider only USA)
data_copy["employee_residence"] = np.where(data["employee_residence"] == "US", 1, 0)
data_copy.rename(columns={"employee_residence": "employee_residence_US"}, inplace=True)

# features: all other variables
features = [
    "work_year",
    "experience_level",
    "employment_type",
    "employee_residence_US",
    "remote_ratio",
    "company_location_US",
    "company_size",
    "modified_job_title",
]
X = data_copy[features]

# response: salary_in_usd
y = data_copy["salary_in_usd"]

# split the data into training and validation set
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, random_state=0
)

# select categorical columns
cat_col = data_copy.select_dtypes(include=["object", "category"]).columns
cat_col = cat_col.drop(["salary_currency", "job_title", "ISO-3_emp", "ISO-3_comp"])

# select numerical columns
num_col = data_copy.select_dtypes(include=["float", "int"]).columns
num_col = num_col.drop(["salary", "salary_in_usd"])

# preprocessing steps for numerical data
num_transformer = SimpleImputer(strategy="median")

# preprocessing steps for categorical data
cat_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("hotencoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# combine both of them
preprocessor = ColumnTransformer(
    transformers=[("num", num_transformer, num_col), ("cat", cat_transformer, cat_col)]
)

# define the model
model = RandomForestRegressor(random_state=0)

######################
# RANDOM FOREST REGRESSION
######################
# build the pipeline
rf_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("rf", model)])

# fit the model
rf_pipeline.fit(X_train, y_train)

# get prediction
rf_pred = rf_pipeline.predict(X_valid)

# mean absolute error
rf_MAE = round(mean_absolute_error(y_valid, rf_pred), 2)

#  r2 score
rf_r2 = round(r2_score(y_valid, rf_pred), 2)

# hyperparameter tuning
rf_param_grid = {
    "rf__n_estimators": [50, 100, 200],
    "rf__max_depth": [None, 5, 10],
    "rf__min_samples_split": [2, 5],
    "rf__min_samples_leaf": [1, 2],
}
# define the Gridsearch object
rf_grid_search = GridSearchCV(rf_pipeline, rf_param_grid, cv=5)

# fit the Gridsearch object
rf_grid_search.fit(X_train, y_train)

# cross validation score
rf_CV = round(rf_grid_search.best_score_, 2)

# validation set score
rf_valid_score = round(rf_grid_search.score(X_valid, y_valid), 2)

print(
    "Best parameters (RandomForestRegressor): {} \n".format(rf_grid_search.best_params_)
)


######################
# DECISION TREE REGRESSOR
######################
# define the model
dr = DecisionTreeRegressor(random_state=0)

# build the pipeline
dr_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("dr", dr)])

# fit the model
dr_pipeline.fit(X_train, y_train)

# get prediction
dr_pred = dr_pipeline.predict(X_valid)

# mean absolute error
dr_MAE = round(mean_absolute_error(y_valid, dr_pred), 2)

#  r2 score
dr_r2 = round(r2_score(y_valid, dr_pred), 2)

# hyperparameter tuning
dr_param_grid = {
    "dr__splitter": ["best", "random"],
    "dr__max_depth": [1, 3, 5],
    "dr__min_samples_leaf": [1, 2, 3, 4, 5],
    "dr__min_weight_fraction_leaf": [0.1, 0.2, 0.3, 0.4, 0.5],
    "dr__max_features": ["auto", "log2", "sqrt", None],
    "dr__max_leaf_nodes": [None, 10, 20, 30, 40, 50],
}
# define the Gridsearch object
dr_grid_search = GridSearchCV(dr_pipeline, dr_param_grid, cv=5)

# fit the Gridsearch object
dr_grid_search.fit(X_train, y_train)

# cross validation score
dr_CV = round(dr_grid_search.best_score_, 2)

# validation set score
dr_valid_score = round(dr_grid_search.score(X_valid, y_valid), 2)

print(
    "Best parameters (DecisionTreeRegressor): {} \n".format(dr_grid_search.best_params_)
)

######################
# GRADIENT BOOSTING REGRESSOR
######################
# define the model
gbr = GradientBoostingRegressor(random_state=0)

# build the pipeline
gbr_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("gbr", gbr)])

# fit the model
gbr_pipeline.fit(X_train, y_train)

# get prediction
gbr_pred = gbr_pipeline.predict(X_valid)

# mean absolute error
gbr_MAE = round(mean_absolute_error(y_valid, gbr_pred), 2)

#  r2 score
gbr_r2 = round(r2_score(y_valid, gbr_pred), 2)

# hyperparameter tuning
gbr_param_grid = {
    "gbr__learning_rate": [0.1, 0.05, 0.01],
    "gbr__n_estimators": [50, 100, 200],
    "gbr__max_depth": [3, 5, 7],
    "gbr__min_samples_split": [2, 4, 8],
    "gbr__min_samples_leaf": [1, 2, 4],
    "gbr__max_features": ["auto", "sqrt", "log2"],
}
# define the Gridsearch object
gbr_grid_search = GridSearchCV(gbr_pipeline, gbr_param_grid, cv=5)

# fit the Gridsearch object
gbr_grid_search.fit(X_train, y_train)

# cross validation score
gbr_CV = round(gbr_grid_search.best_score_, 2)

# validation set score
gbr_valid_score = round(gbr_grid_search.score(X_valid, y_valid), 2)

print(
    "Best parameters (GradientBoostingRegressor): {} \n".format(
        gbr_grid_search.best_params_
    )
)


######################
# LINEAR REGRESSION
######################
# define the model
lr = LinearRegression()

# build the pipeline
lr_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("lr", lr)])

# fit the model
lr_pipeline.fit(X_train, y_train)

# get prediction
lr_pred = lr_pipeline.predict(X_valid)

# mean absolute error
lr_MAE = round(mean_absolute_error(y_valid, lr_pred), 2)

#  r2 score
lr_r2 = round(r2_score(y_valid, lr_pred), 2)

# hyperparameter tuning
lr_param_grid = {"lr__fit_intercept": [True, False]}

# define the Gridsearch object
lr_grid_search = GridSearchCV(lr_pipeline, lr_param_grid, cv=5)

# fit the Gridsearch object
lr_grid_search.fit(X_train, y_train)

# Cross Validation score
lr_CV = round(lr_grid_search.best_score_, 2)

# validation set score
lr_valid_score = round(lr_grid_search.score(X_valid, y_valid), 2)

print("Best parameters (LinearRegressor): {} \n".format(lr_grid_search.best_params_))


# validation set score
lr_valid_score = round(lr_grid_search.score(X_valid, y_valid), 2)

######################
# MODEL SCORES
######################
model_scores = {
    "RandomForestRegression": {
        "MAE": rf_MAE,
        "r2 score": rf_r2,
        "CV score": rf_CV,
        "validation set score": rf_valid_score,
    },
    "DecisionTreeRregressor": {
        "MAE": dr_MAE,
        "r2 score": dr_r2,
        "CV score": dr_CV,
        "validation set score": dr_valid_score,
    },
    "GradientBoostingRegressor": {
        "MAE": gbr_MAE,
        "r2 score": gbr_r2,
        "CV score": gbr_CV,
        "validation set score": gbr_valid_score,
    },
    "Linear Regression": {
        "MAE": lr_MAE,
        "r2 score": lr_r2,
        "CV score": lr_CV,
        "validation set score": lr_valid_score,
    },
}
model_scores = pd.DataFrame.from_dict(model_scores, orient="index")
print(model_scores.T)

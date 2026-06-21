# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/AnkitImpetus/mlops-hf-space-data/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
tourism_dataset.drop(columns=['Unnamed: 0','CustomerID'], inplace=True)

# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = [
    'Age',                       # Customer age
    'CityTier',                  # City classification tier
    'DurationOfPitch',           # Duration of sales pitch
    'NumberOfPersonVisiting',    # Number of people visiting
    'NumberOfFollowups',         # Number of follow-up contacts
    'PreferredPropertyStar',     # Preferred hotel/property star rating
    'NumberOfTrips',             # Number of trips taken
    'Passport',                  # Has passport (0/1)
    'PitchSatisfactionScore',    # Satisfaction score after pitch
    'OwnCar',                    # Owns a car (0/1)
    'NumberOfChildrenVisiting',  # Number of children visiting
    'MonthlyIncome'              # Monthly income
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',             # Contact method used
    'Occupation',                # Customer occupation
    'Gender',                    # Customer gender
    'ProductPitched',            # Tourism package pitched
    'MaritalStatus',             # Marital status
    'Designation'                # Customer designation/job level
]

# Define predictor matrix (X) using selected numeric and categorical features
X = tourism_dataset[numeric_features + categorical_features]

# Define target variable
y = tourism_dataset[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="AnkitImpetus/mlops-hf-space-data",
        repo_type="dataset",
    )

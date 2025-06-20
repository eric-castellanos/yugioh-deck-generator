import pandas as pd
import boto3
import io

# Set up the S3 client
s3_client = boto3.client('s3')

# Define the bucket and object key
bucket = 'yugioh-data'
key = 'deck_scoring/training_data/training_random_generated_decks_composite.csv'

# Read the data from S3
obj = s3_client.get_object(Bucket=bucket, Key=key)
df = pd.read_csv(io.BytesIO(obj['Body'].read()))

# Display basic information about the dataframe
print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())
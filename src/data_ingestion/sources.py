import pandas as pd
import boto3
from io import BytesIO
import requests

class DataFetcher:
    @staticmethod
    def from_csv(path, **kwargs):
        return pd.read_csv(path, **kwargs)
    
    @staticmethod
    def from_s3(bucket, key, aws_access_key, aws_secret_key, region):
        session = boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
        s3 = session.client('s3')
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(BytesIO(obj['Body'].read()))
    
    @staticmethod
    def from_api(url, params, headers, format='json'):
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        
        if format == 'json':
            return pd.json_normalize(response.json())
        elif format == 'csv':
            return pd.read_csv(BytesIO(response.content))
        else:
            raise ValueError(f"Unsupported format: {format}")
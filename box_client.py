import boto3
import pickle
from boxsdk import OAuth2, Client

s3_client = boto3.client(
    's3',
    aws_access_key_id="AKIAIXTWNRNPTLMWZA2Q",
    aws_secret_access_key="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"
)

def s3_dump(obj, bucket, key):
    byte_str = pickle.dumps(obj)
    s3_client.put_object(Body=byte_str, Bucket=bucket, Key=key)

def s3_load(bucket, key):
    s3_obj = s3_client.get_object(Bucket=bucket, Key=key)
    return pickle.loads(s3_obj["Body"].read())

def store_tokens(access_token, refresh_token):
    s3_dump(access_token, bucket="visual-seizure-detection", key="stan_access_token.pickle")
    s3_dump(refresh_token, bucket="visual-seizure-detection", key="stan_refresh_token.pickle")

def get_box_access_tokens():
    access_token = s3_load(bucket="visual-seizure-detection", key="stan_access_token.pickle")
    refresh_token = s3_load(bucket="visual-seizure-detection", key="stan_refresh_token.pickle")
    
    return access_token, refresh_token

def get_box_client():
    access_token, refresh_token = get_box_access_tokens()
    
    oauth = OAuth2(client_id = "nzcw2drgf4qrlhirjcjs0efqt61ilull",
                   client_secret = "3dqQWFx4ei0GMErDZkYFJ7V9t3JAXb63",
                   access_token = access_token,
                   refresh_token = refresh_token,
                   store_tokens = store_tokens)
    
    return Client(oauth)
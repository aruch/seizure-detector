import boto3
import pickle
import threading
import os

from boxsdk import OAuth2, Client

class BoxClient:
    def __init__(self, n_workers=8):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id="AKIAIXTWNRNPTLMWZA2Q",
            aws_secret_access_key="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"
        )
        self.n_workers = n_workers
        self.token_lock = threading.Lock()
        self.box_client = self.get_box_client()
        self.worker_shares = threading.Semaphore(n_workers)

    def s3_dump(self, obj, bucket, key):
        byte_str = pickle.dumps(obj)
        self.s3_client.put_object(Body=byte_str, Bucket=bucket, Key=key)

    def s3_load(self, bucket, key):
        s3_obj = self.s3_client.get_object(Bucket=bucket, Key=key)
        return pickle.loads(s3_obj["Body"].read())

    def store_tokens(self, access_token, refresh_token):
        self.token_lock.acquire()
        try:
            self.s3_dump(access_token, bucket="visual-seizure-detection", key="stan_access_token.pickle")
            self.s3_dump(refresh_token, bucket="visual-seizure-detection", key="stan_refresh_token.pickle")
        finally:
            self.token_lock.release()

    def download_file(self, file_id, output_dir):
        print("Downloading {:s}".format(file_id))
        try:
            box_file = self.box_client.file(file_id)
            with open(os.path.join(output_dir, box_file.get().name), "wb") as output_file:
                box_file.download_to(output_file)
        finally:
            self.worker_shares.release()

    def download_files(self, file_ids, output_dir):
        download_threads = []
        for file_id in file_ids:
            self.worker_shares.acquire()
            t = threading.Thread(target=self.download_file, args=(file_id, output_dir))
            t.start()
            download_threads.append(t)
        for t in download_threads:
            t.join()

    def get_box_access_tokens(self):
        access_token = self.s3_load(bucket="visual-seizure-detection", key="stan_access_token.pickle")
        refresh_token = self.s3_load(bucket="visual-seizure-detection", key="stan_refresh_token.pickle")

        return access_token, refresh_token

    def get_box_client(self):
        access_token, refresh_token = self.get_box_access_tokens()

        oauth = OAuth2(client_id = "nzcw2drgf4qrlhirjcjs0efqt61ilull",
                       client_secret = "3dqQWFx4ei0GMErDZkYFJ7V9t3JAXb63",
                       access_token = access_token,
                       refresh_token = refresh_token,
                       store_tokens = self.store_tokens)

        return Client(oauth)
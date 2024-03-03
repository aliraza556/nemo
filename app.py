from flask import Flask
import os
import urllib3
import boto3
import json
import time

from api.service.diarization_service import DiarizationService
from api.service.download_service import DownloadService


def create_diarization_service():
    download_service = DownloadService()
    return DiarizationService(download_service)

def batch():
    input_file_url = os.environ['INPUT_FILE_URL']
    http = urllib3.PoolManager()
    
    try:
        diarization_service = create_diarization_service()
        result = diarization_service.call_nemo(input_file_url)
        body = json.dumps({'status': 'success', 'data': result}).encode('utf-8')
    except Exception as e:
        body = json.dumps({'status': 'error', 'message': f"AWS Batch Job Error: {e}"}).encode('utf-8')

    print(body)


if __name__ == '__main__':
    batch()

import requests
import base64
import hashlib

FAMILY_NAME = "simplechain"
REST_API = "http://localhost:8008"

def make_address(key):
    namespace = hashlib.sha512(FAMILY_NAME.encode()).hexdigest()[0:6]
    return namespace + hashlib.sha512(key.encode()).hexdigest()[0:64]


def read(key):
    address = make_address(key)
    r = requests.get(f"{REST_API}/state/{address}")
    data = base64.b64decode(r.json()["data"]).decode()
    print("Value:", data)


read("name")

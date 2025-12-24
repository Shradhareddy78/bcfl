import sys
import hashlib
import json
from sawtooth_sdk.processor.core import TransactionProcessor
from sawtooth_sdk.processor.handler import TransactionHandler
from sawtooth_sdk.processor.exceptions import InvalidTransaction
from sawtooth_sdk.processor.state import StateEntry

FAMILY_NAME = "modelupdates"
FAMILY_VERSION = "1.0"

def make_address(round_number):
    prefix = hashlib.sha512(FAMILY_NAME.encode()).hexdigest()[0:6]
    return prefix + hashlib.sha512(str(round_number).encode()).hexdigest()[-64:]

class ModelUpdateHandler(TransactionHandler):
    def __init__(self):
        super().__init__(FAMILY_NAME, FAMILY_VERSION, [FAMILY_NAME])

    def apply(self, transaction, context):
        payload = json.loads(transaction.payload.decode())

        round_number = payload["round"]
        address = make_address(round_number)

        print(f"📌 Processing Round: {round_number}")
        print(f"Writing to Address: {address}")

        data_bytes = json.dumps(payload).encode()
        state_entry = StateEntry(address=address, data=data_bytes)
        context.set_state({address: data_bytes})

if __name__ == "__main__":
    processor = TransactionProcessor(url="tcp://localhost:4004")
    handler = ModelUpdateHandler()
    processor.add_handler(handler)
    processor.start()

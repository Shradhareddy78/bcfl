import hashlib
import json
from sawtooth_sdk.processor.core import TransactionProcessor
from sawtooth_sdk.processor.handler import TransactionHandler

FAMILY_NAME = "flchain"
FAMILY_VERSION = "1.0"
NAMESPACE = hashlib.sha512(FAMILY_NAME.encode()).hexdigest()[:6]

def make_address(key):
    return NAMESPACE + hashlib.sha512(key.encode()).hexdigest()[:64]

class FLChainHandler(TransactionHandler):
    @property
    def family_name(self):
        return FAMILY_NAME

    @property
    def family_versions(self):
        return [FAMILY_VERSION]

    @property
    def namespaces(self):
        return [NAMESPACE]

    def apply(self, txn, context):
        payload = json.loads(txn.payload.decode())
        addr = make_address("round_" + str(payload["round"]))
        context.set_state({addr: txn.payload})

if __name__ == "__main__":
    tp = TransactionProcessor("tcp://validator:4004")
    tp.add_handler(FLChainHandler())
    tp.start()

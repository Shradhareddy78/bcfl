import hashlib
import sys
import logging

from sawtooth_sdk.processor.core import TransactionProcessor
from sawtooth_sdk.processor.handler import TransactionHandler
from sawtooth_sdk.processor.exceptions import InvalidTransaction

LOGGER = logging.getLogger(__name__)

FAMILY_NAME = "simplechain"
FAMILY_VERSION = "1.0"
NAMESPACE = hashlib.sha512(FAMILY_NAME.encode()).hexdigest()[0:6]


def make_address(key):
    return NAMESPACE + hashlib.sha512(key.encode()).hexdigest()[0:64]


class SimpleHandler(TransactionHandler):
    @property
    def family_name(self):
        return FAMILY_NAME

    @property
    def family_versions(self):
        return [FAMILY_VERSION]

    @property
    def namespaces(self):
        return [NAMESPACE]

    def apply(self, transaction, context):
        payload = transaction.payload.decode()
        LOGGER.info("Payload received: %s", payload)

        if ":" not in payload:
            raise InvalidTransaction("Payload must be key:value")

        key, value = payload.split(":", 1)
        address = make_address(key)

        state_entries = context.get_state([address])
        data = value.encode()

        context.set_state({address: data})


def main():
    logging.basicConfig(level=logging.INFO)
    processor = TransactionProcessor(url="tcp://localhost:4004")
    handler = SimpleHandler()
    processor.add_handler(handler)
    processor.start()


if __name__ == "__main__":
    main()

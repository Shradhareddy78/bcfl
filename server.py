import flwr as fl
import hashlib
import json
import time
import random
import requests
import numpy as np
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

from sawtooth_signing import create_context, CryptoFactory
from sawtooth_sdk.protobuf.transaction_pb2 import TransactionHeader, Transaction
from sawtooth_sdk.protobuf.batch_pb2 import BatchHeader, Batch, BatchList

# ======================================
# SAWTOOTH CONFIG (FIXED)
# ======================================
FAMILY_NAME = "flchain"
FAMILY_VERSION = "1.0"

NAMESPACE = hashlib.sha512(FAMILY_NAME.encode()).hexdigest()[:6]
SAWTOOTH_REST_API = "http://host.docker.internal:8008/batches"
ROUND_COUNTER = 0

# ======================================
# ADDRESS HELPER (CRITICAL)
# ======================================
def make_address(key: str) -> str:
    return NAMESPACE + hashlib.sha512(key.encode()).hexdigest()[:64]

# ======================================
# SIMULATED LEADER ELECTION
# ======================================
def elect_leader(client_ids):
    time.sleep(random.uniform(0.2, 1.0))
    return random.choice(client_ids)

# ======================================
# SUBMIT TO SAWTOOTH
# ======================================
def submit_to_sawtooth(payload_dict):
    payload_bytes = json.dumps(payload_dict).encode()

    context = create_context("secp256k1")
    private_key = context.new_random_private_key()
    signer = CryptoFactory(context).new_signer(private_key)

    address = make_address("round_" + str(payload_dict["round"]))

    txn_header = TransactionHeader(
        family_name=FAMILY_NAME,
        family_version=FAMILY_VERSION,
        inputs=[address],
        outputs=[address],
        signer_public_key=signer.get_public_key().as_hex(),
        batcher_public_key=signer.get_public_key().as_hex(),
        dependencies=[],
        payload_sha512=hashlib.sha512(payload_bytes).hexdigest(),
    ).SerializeToString()

    transaction = Transaction(
        header=txn_header,
        payload=payload_bytes,
        header_signature=signer.sign(txn_header),
    )

    batch_header = BatchHeader(
        signer_public_key=signer.get_public_key().as_hex(),
        transaction_ids=[transaction.header_signature],
    ).SerializeToString()

    batch = Batch(
        header=batch_header,
        transactions=[transaction],
        header_signature=signer.sign(batch_header),
    )

    batch_list = BatchList(batches=[batch])

    # 🔒 FAULT-TOLERANT SAWTOOTH SUBMISSION
    try:
        r = requests.post(
            SAWTOOTH_REST_API,
            headers={"Content-Type": "application/octet-stream"},
            data=batch_list.SerializeToString(),
            timeout=5
        )
        print("⛓️  Sawtooth status:", r.status_code)
    except Exception as e:
        print("⚠️ Sawtooth unavailable, skipping blockchain write:", e)

# ======================================
# FLOWER STRATEGY
# ======================================
class Strategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        global ROUND_COUNTER

        aggregated, metrics = super().aggregate_fit(rnd, results, failures)
        ROUND_COUNTER += 1

        client_ids = [res[0].cid for res in results]

        if aggregated and client_ids:
            ndarrays = parameters_to_ndarrays(aggregated)
            model_hash = hashlib.sha256(ndarrays[0].tobytes()).hexdigest()

            payload = {
                "round": ROUND_COUNTER,
                "model_hash": model_hash,
                "clients": client_ids,
                "leader": elect_leader(client_ids),
                "timestamp": time.time(),
            }

            print(f"\n🔐 FL Round {ROUND_COUNTER}")
            print(json.dumps(payload, indent=2))

            submit_to_sawtooth(payload)

        return aggregated, metrics

# ======================================
# INITIAL PARAMETERS (REQUIRED)
# ======================================
initial_parameters = ndarrays_to_parameters(
    [np.zeros(1, dtype=np.float32)]
)

# ======================================
# START SERVER
# ======================================
if __name__ == "__main__":
    print("🚀 Flower Server + Sawtooth logging")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=Strategy(initial_parameters=initial_parameters),
        config=fl.server.ServerConfig(num_rounds=3),
    )

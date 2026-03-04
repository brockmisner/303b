"""
One-time CTF Exchange approval script.

Grants infinite global approval to the Polymarket CTF Exchange contract
so your bot can SELL outcome tokens (UP/DOWN) without per-trade approvals.

Run ONCE:
    python approve_ctf.py

After this confirms on Polygon, you will never see the
"not enough balance / allowance" 400 error again.

Requirements:
    pip install web3 python-dotenv
"""

import os
import sys
import time
from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

# ── Polygon Mainnet Config ──────────────────────────────────────────────
PRIVATE_KEY = os.getenv("CLOB_PRIVATE_KEY") or os.getenv("PK")
ALCHEMY_KEY = os.getenv("ALCHEMY_API_KEY")
RPC_URL = (
    f"https://polygon-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}"
    if ALCHEMY_KEY
    else os.getenv("POLYGON_RPC", "https://polygon-rpc.com")
)

# ── Contract Addresses (Polygon Mainnet) ────────────────────────────────
USDC_ADDRESS      = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
CTF_ADDRESS       = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
CTF_EXCHANGE      = Web3.to_checksum_address("0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E")
NEG_RISK_EXCHANGE = Web3.to_checksum_address("0xC5d563A36AE78145C45a50134d48A1215220f80a")
NEG_RISK_ADAPTER  = Web3.to_checksum_address("0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296")

MAX_UINT256 = 2**256 - 1

# ── ABIs (minimal) ──────────────────────────────────────────────────────
ERC20_APPROVE_ABI = [{
    "inputs": [
        {"name": "spender", "type": "address"},
        {"name": "amount", "type": "uint256"},
    ],
    "name": "approve",
    "outputs": [{"name": "", "type": "bool"}],
    "stateMutability": "nonpayable",
    "type": "function",
}]

ERC1155_SET_APPROVAL_ABI = [{
    "inputs": [
        {"name": "operator", "type": "address"},
        {"name": "approved", "type": "bool"},
    ],
    "name": "setApprovalForAll",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function",
}]


def main():
    if not PRIVATE_KEY or PRIVATE_KEY == "YOUR_PRIVATE_KEY":
        print("ERROR: Set CLOB_PRIVATE_KEY or PK in your .env file.")
        sys.exit(1)

    w3 = Web3(Web3.HTTPProvider(RPC_URL))

    # Inject PoA middleware for Polygon
    try:
        from web3.middleware import ExtraDataToPOAMiddleware
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    except ImportError:
        try:
            from web3.middleware import ExtraDataToPoa
            w3.middleware_onion.inject(ExtraDataToPoa, layer=0)
        except ImportError:
            from web3.middleware import geth_poa_middleware
            w3.middleware_onion.inject(geth_poa_middleware, layer=0)

    if not w3.is_connected():
        print(f"ERROR: Cannot connect to RPC at {RPC_URL}")
        sys.exit(1)

    account = w3.eth.account.from_key(PRIVATE_KEY)
    wallet = account.address
    print(f"Wallet:  {wallet}")
    print(f"RPC:     {RPC_URL[:60]}...")
    print(f"Balance: {w3.eth.get_balance(wallet) / 1e18:.4f} MATIC")
    print()

    usdc_contract = w3.eth.contract(address=USDC_ADDRESS, abi=ERC20_APPROVE_ABI)
    ctf_contract  = w3.eth.contract(address=CTF_ADDRESS, abi=ERC1155_SET_APPROVAL_ABI)

    approvals = [
        # 1. USDC -> CTF Exchange (spend USDC to buy tokens)
        ("USDC approve -> CTF Exchange", usdc_contract.functions.approve(CTF_EXCHANGE, MAX_UINT256)),
        # 2. USDC -> Neg Risk Exchange
        ("USDC approve -> NegRisk Exchange", usdc_contract.functions.approve(NEG_RISK_EXCHANGE, MAX_UINT256)),
        # 3. CTF tokens -> CTF Exchange (sell outcome tokens)
        ("CTF setApprovalForAll -> CTF Exchange", ctf_contract.functions.setApprovalForAll(CTF_EXCHANGE, True)),
        # 4. CTF tokens -> Neg Risk Exchange
        ("CTF setApprovalForAll -> NegRisk Exchange", ctf_contract.functions.setApprovalForAll(NEG_RISK_EXCHANGE, True)),
        # 5. CTF tokens -> Neg Risk Adapter
        ("CTF setApprovalForAll -> NegRisk Adapter", ctf_contract.functions.setApprovalForAll(NEG_RISK_ADAPTER, True)),
    ]

    for label, tx_fn in approvals:
        print(f"  Sending: {label} ...")
        try:
            nonce = w3.eth.get_transaction_count(wallet)
            tx = tx_fn.build_transaction({
                "from": wallet,
                "nonce": nonce,
                "gas": 80_000,
                "maxFeePerGas": w3.to_wei(50, "gwei"),
                "maxPriorityFeePerGas": w3.to_wei(30, "gwei"),
            })
            signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            print(f"    TX: {tx_hash.hex()}")
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            status = "SUCCESS" if receipt["status"] == 1 else "FAILED"
            print(f"    {status} (block {receipt['blockNumber']}, gas {receipt['gasUsed']})")
            if receipt["status"] != 1:
                print(f"    WARNING: Transaction reverted! Check on Polygonscan.")
        except Exception as e:
            print(f"    ERROR: {e}")
        time.sleep(1)  # brief pause between txns

    print()
    print("Done! All approvals submitted.")
    print("Your bot can now BUY and SELL tokens via the CLOB API without allowance errors.")


if __name__ == "__main__":
    main()

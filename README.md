# LlamaIndex Tools Integration: InsumerAPI

Wallet auth and condition-based access for LlamaIndex agents. Across 38 chains — read → evaluate → sign, returning an ECDSA-signed boolean your agent can verify offline against the public JWKS. Boolean, not balance: the API never exposes wallet holdings, only a signed yes-or-no against the conditions you configure.

Part of [InsumerAPI](https://insumermodel.com/developers/). No secrets. No identity-first. No static credentials.

## Installation

```bash
pip install llama-index-tools-insumer
# the Quickstart's agent additionally needs:
pip install llama-index-agent-openai
```

## Quickstart

**Get a key — no signup, no dashboard, no password.** Two paths, both return an `insr_live_...` key instantly with 10 verification credits and 100 reads/day:

```bash
curl -X POST https://api.insumermodel.com/v1/keys/create \
    -H "Content-Type: application/json" \
    -d '{"email": "you@example.com", "appName": "my-agent", "tier": "free"}'
```

Or enter your email on [insumermodel.com](https://insumermodel.com/?utm_source=pypi-llama-index-tools-insumer) — the key appears inline. Already have a key? Manage it at [insumermodel.com/developers/account/](https://insumermodel.com/developers/account/?utm_source=pypi-llama-index-tools-insumer).

Then use the tool spec in any LlamaIndex agent:

```python
from llama_index.tools.insumer import InsumerToolSpec
from llama_index.agent.openai import OpenAIAgent

insumer = InsumerToolSpec()  # reads INSUMER_API_KEY; or pass api_key="insr_live_..."

agent = OpenAIAgent.from_tools(
    insumer.to_tool_list(),
    verbose=True,
)

agent.chat(
    "Does wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045 "
    "hold at least 100 USDC on Base?"
)
```

## The six tools

### `attest_wallet`

Run wallet attestation against 1–10 conditions. Returns an ECDSA-signed verdict per condition plus a condition hash for tamper detection.

Supported condition types:

- `token_balance` — ERC-20 / SPL / XRPL trust line / native BTC / TRC-20 / Stellar trustline / Sui-native ≥ threshold
- `nft_ownership` — ERC-721/ERC-1155/XRPL NFToken holding
- `eas_attestation` — EAS schema check (pass a `template` like `coinbase_verified_account` or a raw `schemaId`)
- `farcaster_id` — Farcaster ID registered on Optimism
- `ratio_to_amount` — self-scaling agent-spend rule: balance ≥ `multiple` × `amount` (RPC EVM chains only)
- `ratio_to_supply` — share-of-supply rule: balance / `totalSupply()` ≥ `minFraction`, a fraction in (0, 1] (RPC EVM chains, ERC-20 only)

```python
insumer.attest_wallet(
    wallet="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
    conditions=[
        {
            "type": "token_balance",
            "contractAddress": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            "chainId": 8453,
            "threshold": "100",
            "decimals": 6,
            "label": "USDC on Base >= 100",
        },
    ],
)
```

> **`token_balance` thresholds are decimal strings** — send `"threshold": "100"`, not `100`. Keys created from 2026-06-10 sign with `kid: insumer-attest-v2` and reject a JSON number with a `400` (a string works on both v1 and v2 keys). This tool coerces a number to a string for you.

Response shape:

```python
{
    "ok": True,
    "data": {
        "attestation": {
            "id": "ATST-...",
            "pass": True,
            "results": [...],      # per-condition booleans + conditionHash
            "passCount": 1,
            "failCount": 0,
            "attestedAt": "2026-04-16T...",
            "expiresAt": "2026-04-16T...",  # +30 min
        },
        "sig": "...",              # ECDSA P-256 signature, base64
        "kid": "insumer-attest-v2",   # keys minted today; pre-cutover keys return insumer-attest-v1
    },
    "meta": {"creditsRemaining": ..., "creditsCharged": 1, ...},
}
```

Costs 1 credit per call (2 with `proof="merkle"` for EIP-1186 storage proofs).

### `get_trust_profile`

Multi-dimensional wallet trust profile — stablecoins, governance, NFTs, staking (plus Solana, XRPL, Bitcoin, Tron, Stellar, Sui when those wallet addresses are supplied). Returns a signed summary showing which dimensions have activity, without exposing raw balances. Up to 49 checks across 27 chains.

```python
insumer.get_trust_profile(
    wallet="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
    solana_wallet="...",       # optional
    xrpl_wallet="r...",        # optional
    bitcoin_wallet="bc1q...",  # optional
    tron_wallet="T...",        # optional
    stellar_wallet="G...",     # optional
    sui_wallet="0x...",        # optional (64 hex chars)
)
```

Costs 3 credits per call (6 with `proof="merkle"`).

### `list_compliance_templates`

Discover pre-configured EAS compliance templates (Coinbase Verified Account, Coinbase Verified Country, Coinbase One, Gitcoin Passport, etc.). No API key required.

```python
templates = insumer.list_compliance_templates()
# Use a template name directly in attest_wallet:
insumer.attest_wallet(
    wallet="0x...",
    conditions=[{"type": "eas_attestation", "template": "coinbase_verified_account"}],
)
```

### `get_jwks`

Fetch the public JWKS used to sign attestation and trust responses. Enables offline verification of any result with a standard JWT/JOSE library. No API key required.

```python
jwks = insumer.get_jwks()
# {
#     "keys": [
#         {"kty": "EC", "crv": "P-256", "x": "...", "y": "...",
#          "use": "sig", "alg": "ES256", "kid": "insumer-attest-v1"},
#         {..., "kid": "insumer-attest-v2"},   # match on the kid your response carries
#         {..., "kid": "insumer-trust-v2"}
#     ]
# }
```

### `buy_api_key`

Agentic commerce: let an agent purchase its own new API key on-chain with USDC or BTC. The transaction sender wallet becomes the registered identity. No email, no signup flow, no human in the loop. No API key required to call — the payment *is* the auth.

Pre-requisite: broadcast a USDC or BTC transfer to the platform wallet first, then submit the transaction hash here.

```python
# After the agent broadcasts a 100 USDC transfer on Base to the platform wallet:
result = InsumerToolSpec().buy_api_key(
    tx_hash="0xabc...",
    chain_id=8453,            # Base; use "solana", "bitcoin", or "tron" for those chains
    app_name="my-agent",
    amount=100.0,              # USDC amount; not required for Bitcoin
)
new_key = result["data"]["key"]   # insr_live_... — shown once, save it
```

One key per wallet — if the sending wallet already has a self-serve key, the API returns 409 and asks you to top up the existing key via `buy_credits` instead.

### `buy_credits`

Top up credits on an existing API key (the one you passed to `InsumerToolSpec`) with a USDC or BTC payment. Same pattern: broadcast the transfer, submit the `tx_hash`.

```python
insumer.buy_credits(
    tx_hash="0xdef...",
    chain_id=8453,
    amount=100.0,
)
```

## No key at all: x402 pay-per-call

This tool spec authenticates with an API key (free tier: 10 credits via `POST /v1/keys/create`, or on-chain purchase via `buy_api_key`). If your agent wants zero signup of any kind, the same core endpoints — `POST /v1/attest`, `POST /v1/trust`, `POST /v1/trust/batch` — also accept [x402](https://www.x402.org) payments directly: call with no credentials, receive a 402 quote, sign an EIP-3009 USDC authorization on Base for the exact quoted amount, retry with the `X-PAYMENT` header. $0.05 per attestation ($0.10 with a Merkle proof), settlement is gasless for the payer. That flow lives outside this package (it needs wallet signing, not an LLM tool), but the responses are identical — same signed attestations, verifiable against the same JWKS.

Pay-per-call wallets that reach $1.00 in cumulative spend become eligible to claim a soulbound Insumer Access pass (nothing mints unless the wallet asks — its first wallet-auth request is the claim), which unlocks prepaid credit rates ($0.04–$0.02/call) and single-round-trip calls. Details: [insumermodel.com/llms.txt](https://insumermodel.com/llms.txt).

## Supported chains

38 total:

- **32 EVM chains**: Ethereum, Base, Arbitrum, Optimism, Polygon, Avalanche, BNB, XDC, Robinhood Chain, Unichain, Linea, zkSync, Scroll, Blast, Mantle, Celo, Gnosis, Sonic, Moonbeam, and more
- **Solana** (mainnet)
- **XRPL** (mainnet) — native XRP plus trust-line tokens
- **Bitcoin** (mainnet) — native BTC only
- **Tron** — native TRX plus TRC-20 (USDT-TRC20)
- **Stellar** — native XLM plus classic trustline assets (USDC, BENJI, etc.)
- **Sui** — native SUI plus Sui-native tokens (USDC)

## Positioning

Wallet auth is the primitive. Condition-based access is the category. Token gating is one use case. The API turns a programmable predicate over on-chain state into a short-lived cryptographic artifact any service can verify.

- **No secrets**: conditions are public, the signature binds the condition hash.
- **No identity-first**: a wallet address and a condition are enough.
- **No static credentials**: every response has an expiry and is re-checkable.

## Learn more

- Docs: [insumermodel.com/developers/](https://insumermodel.com/developers/)
- OpenAPI spec: [insumermodel.com/openapi.yaml](https://insumermodel.com/openapi.yaml)
- Public JWKS: [api.insumermodel.com/.well-known/jwks.json](https://api.insumermodel.com/.well-known/jwks.json)
- Companion packages: `langchain-insumer` (LangChain), `mcp-server-insumer` (Model Context Protocol), `eliza-plugin-insumer` (ElizaOS)

## License

Apache-2.0

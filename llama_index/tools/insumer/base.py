"""InsumerAPI tool spec for LlamaIndex.

Wallet auth and condition-based access across 33 chains.
Read --> evaluate --> sign. Returns an ECDSA-signed boolean you can verify
offline against our public JWKS. Boolean, not balance: the API never exposes
wallet holdings, only a signed yes-or-no against the conditions you configure.
"""

from typing import Any, Dict, List, Optional

import requests
from llama_index.core.tools.tool_spec.base import BaseToolSpec

DEFAULT_BASE_URL = "https://api.insumermodel.com"
DEFAULT_JWKS_URL = "https://api.insumermodel.com/.well-known/jwks.json"
DEFAULT_TIMEOUT = 30


class InsumerToolSpec(BaseToolSpec):
    """Tool spec for InsumerAPI.

    Exposes four methods as LlamaIndex tools:

    - ``attest_wallet``: run wallet attestation against one or more conditions
      (token balance, NFT ownership, EAS attestation, Farcaster ID) across
      33 chains. Returns an ECDSA-signed boolean verdict per condition
      plus condition hashes for tamper detection.
    - ``get_trust_profile``: fetch a multi-dimensional wallet trust profile
      (stablecoins, governance, NFTs, staking, plus optional Solana/XRPL/Bitcoin
      dimensions). Returns a signed summary of which dimensions show activity.
    - ``list_compliance_templates``: discover pre-configured compliance
      templates (Coinbase Verified Account, Gitcoin Passport, etc.) usable
      directly in attest_wallet without raw EAS schema IDs. No API key
      required.
    - ``get_jwks``: fetch the JSON Web Key Set used to verify ECDSA signatures
      on attestation and trust responses. No API key required. Enables offline
      verification.
    - ``buy_api_key``: let an agent purchase its own new API key on-chain with
      USDC or BTC, no human in the loop. Wallet address is the identity; no
      email required.
    - ``buy_credits``: top up credits on an existing API key with a USDC or
      BTC payment, no out-of-band billing.
    """

    spec_functions = [
        "attest_wallet",
        "get_trust_profile",
        "list_compliance_templates",
        "get_jwks",
        "buy_api_key",
        "buy_credits",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize the InsumerAPI tool spec.

        Args:
            api_key: Your InsumerAPI key (format ``insr_live_...``). Required
                for ``attest_wallet`` and ``get_trust_profile``. Not needed for
                ``list_compliance_templates`` or ``get_jwks``. Get a free key
                at https://insumermodel.com/developers/.
            base_url: API base URL. Defaults to ``https://api.insumermodel.com``.
            timeout: HTTP request timeout in seconds. Defaults to 30.
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _headers(self, include_auth: bool = True) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if include_auth:
            if not self.api_key:
                raise ValueError(
                    "InsumerAPI key required for this operation. "
                    "Get one at https://insumermodel.com/developers/."
                )
            headers["X-API-Key"] = self.api_key
        return headers

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}{path}",
            json=body,
            headers=self._headers(include_auth=True),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _get(self, path: str, include_auth: bool = False) -> Dict[str, Any]:
        response = requests.get(
            f"{self.base_url}{path}",
            headers=self._headers(include_auth=include_auth),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def attest_wallet(
        self,
        conditions: List[Dict[str, Any]],
        wallet: Optional[str] = None,
        solana_wallet: Optional[str] = None,
        xrpl_wallet: Optional[str] = None,
        bitcoin_wallet: Optional[str] = None,
        proof: Optional[str] = None,
        format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run wallet attestation against 1-10 conditions. Returns an
        ECDSA-signed verdict per condition.

        Wallet auth primitive: read --> evaluate --> sign. The API reads the
        relevant wallet state (token balance, NFT ownership, EAS attestation,
        or Farcaster ID), evaluates it against the caller-specified condition,
        and returns a signed boolean. Raw balances are never returned in
        standard mode (use proof="merkle" if you want the storage proof,
        which reveals the balance).

        Args:
            conditions: List of 1 to 10 condition objects. Each object must
                have a ``type`` field: ``token_balance``, ``nft_ownership``,
                ``eas_attestation``, or ``farcaster_id``. Token balance
                conditions require ``contractAddress``, ``chainId``,
                ``threshold``, and (for EVM) ``decimals``. EAS conditions
                can use a pre-configured ``template`` (from
                list_compliance_templates) or a raw ``schemaId``.
            wallet: EVM wallet address (0x + 40 hex). Required if any
                condition targets an EVM chain.
            solana_wallet: Solana wallet address (base58, 32-44 chars).
                Required for conditions with ``chainId: "solana"``.
            xrpl_wallet: XRPL address (r-address, 25-35 chars). Required for
                conditions with ``chainId: "xrpl"``.
            bitcoin_wallet: Bitcoin address (P2PKH, P2SH, bech32, or Taproot).
                Required for conditions with ``chainId: "bitcoin"``. Bitcoin
                only supports ``token_balance`` with ``contractAddress:
                "native"``.
            proof: Set to ``"merkle"`` to include EIP-1186 Merkle storage
                proofs in results. Costs 2 credits instead of 1. Reveals raw
                balance to the caller.
            format: Set to ``"jwt"`` to include an ES256-signed JWT in the
                response alongside the boolean result.

        Returns:
            API response envelope. On success:

            .. code-block:: python

                {
                    "ok": True,
                    "data": {
                        "attestation": {
                            "id": "ATST-...",
                            "pass": bool,
                            "results": [...],
                            "passCount": int,
                            "failCount": int,
                            "attestedAt": ISO8601,
                            "expiresAt": ISO8601,
                        },
                        "sig": str,                # ECDSA P-256, base64
                        "kid": "insumer-attest-v1",
                        "jwt": str,                # if format="jwt"
                    },
                    "meta": {"creditsRemaining": int, "creditsCharged": int, ...},
                }
        """
        body: Dict[str, Any] = {"conditions": conditions}
        if wallet:
            body["wallet"] = wallet
        if solana_wallet:
            body["solanaWallet"] = solana_wallet
        if xrpl_wallet:
            body["xrplWallet"] = xrpl_wallet
        if bitcoin_wallet:
            body["bitcoinWallet"] = bitcoin_wallet
        if proof:
            body["proof"] = proof
        if format:
            body["format"] = format
        return self._post("/v1/attest", body)

    def get_trust_profile(
        self,
        wallet: str,
        solana_wallet: Optional[str] = None,
        xrpl_wallet: Optional[str] = None,
        bitcoin_wallet: Optional[str] = None,
        proof: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch a multi-dimensional wallet trust profile. Returns an
        ECDSA-signed summary across stablecoins, governance, NFTs, and
        staking dimensions (plus Solana/XRPL/Bitcoin when those wallet
        addresses are provided).

        Trust profile reports which dimensions show activity. Each dimension
        runs a curated set of token/NFT balance checks (``balance > 0``).
        The response includes per-check booleans, a summary, and a signature
        over the whole payload. 3 credits standard, 6 with proof="merkle".

        Args:
            wallet: EVM wallet address to profile (required, 0x + 40 hex).
            solana_wallet: Optional Solana address. Adds Solana USDC check.
            xrpl_wallet: Optional XRPL r-address. Adds XRPL stablecoin checks
                (RLUSD, USDC).
            bitcoin_wallet: Optional Bitcoin address. Adds native BTC balance
                check.
            proof: Set to ``"merkle"`` for EIP-1186 Merkle storage proofs on
                stablecoin/governance checks. Costs 6 credits instead of 3.

        Returns:
            API response envelope. On success:

            .. code-block:: python

                {
                    "ok": True,
                    "data": {
                        "trust": {
                            "id": "TRST-...",
                            "wallet": "0x...",
                            "conditionSetVersion": "v1",
                            "dimensions": {
                                "stablecoins": {"checks": [...], "passCount": int, "failCount": int, "total": int},
                                "governance": {...},
                                "nfts": {...},
                                "staking": {...},
                                # Optional dimensions when wallet addresses provided:
                                "solana": {...},
                                "xrpl": {...},
                                "bitcoin": {...},
                            },
                            "summary": {
                                "totalChecks": int,
                                "totalPassed": int,
                                "totalFailed": int,
                                "dimensionsWithActivity": int,
                                "dimensionsChecked": int,
                            },
                            "profiledAt": ISO8601,
                            "expiresAt": ISO8601,
                        },
                        "sig": str,                # ECDSA P-256, base64
                        "kid": "insumer-attest-v1",
                    },
                    "meta": {"creditsRemaining": int, "creditsCharged": int, ...},
                }
        """
        body: Dict[str, Any] = {"wallet": wallet}
        if solana_wallet:
            body["solanaWallet"] = solana_wallet
        if xrpl_wallet:
            body["xrplWallet"] = xrpl_wallet
        if bitcoin_wallet:
            body["bitcoinWallet"] = bitcoin_wallet
        if proof:
            body["proof"] = proof
        return self._post("/v1/trust", body)

    def list_compliance_templates(self) -> Dict[str, Any]:
        """Discover pre-configured compliance templates for EAS attestations.

        Templates abstract away raw EAS schema IDs, attester addresses, and
        decoder contracts. Pass the template name directly as
        ``conditions[].template`` in attest_wallet.

        No API key required. Response is cached for 1 hour at the edge.

        Returns:
            API response envelope. On success:

            .. code-block:: python

                {
                    "ok": True,
                    "data": {
                        "templates": {
                            "coinbase_verified_account": {
                                "provider": "Coinbase",
                                "description": "Coinbase Verified Account",
                                "chainId": 8453,
                                "chainName": "Base",
                            },
                            "gitcoin_passport_score": {...},
                            ...
                        }
                    },
                    "meta": {...},
                }
        """
        return self._get("/v1/compliance/templates", include_auth=False)

    def get_jwks(self) -> Dict[str, Any]:
        """Fetch the public JSON Web Key Set for offline verification of
        ECDSA signatures on attestation and trust responses.

        Standard JWKS format. Compatible with any JWT/JOSE library (jose,
        PyJWT, python-jose, etc.). No API key required.

        Returns:
            JWKS response:

            .. code-block:: python

                {
                    "keys": [
                        {
                            "kty": "EC",
                            "crv": "P-256",
                            "x": "...",
                            "y": "...",
                            "use": "sig",
                            "alg": "ES256",
                            "kid": "insumer-attest-v1",
                        }
                    ]
                }
        """
        # JWKS is served at the same origin but at /.well-known/jwks.json
        # rather than /v1/*, so we hit the full URL directly.
        response = requests.get(
            DEFAULT_JWKS_URL if self.base_url == DEFAULT_BASE_URL
            else f"{self.base_url}/.well-known/jwks.json",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def buy_api_key(
        self,
        tx_hash: str,
        chain_id: Any,
        app_name: str,
        amount: Optional[float] = None,
        channel: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Buy a new API key on-chain with USDC or BTC. Agent-friendly: the
        sender wallet address of the transaction becomes the registered
        identity on the new key. No email, no human signup flow.

        Pre-requisite: the agent must have already broadcast a USDC or BTC
        transfer to the platform wallet BEFORE calling this method. The
        transaction hash is then submitted here for on-chain verification.

        One key per wallet — if the sending wallet already has a self-serve
        key, the API returns 409 and asks you to top up the existing key
        with ``buy_credits`` instead.

        Keys from this endpoint have a 30-day expiry and tier ``paid``.

        Args:
            tx_hash: Transaction hash of the USDC or BTC transfer to the
                platform wallet. Must not have been used before.
            chain_id: Either an EVM chain ID (int, e.g. 1, 8453, 10) for USDC
                transfers, the string ``"solana"`` for USDC on Solana, or the
                string ``"bitcoin"`` for BTC.
            app_name: Human-readable name for the key (max 100 chars).
            amount: USDC amount paid (required for all USDC chains). Not
                required for Bitcoin — the USD value is derived from the
                on-chain BTC amount and a price feed.
            channel: Optional tracking tag for the purchase channel.

        Returns:
            API response envelope with the newly issued raw API key:

            .. code-block:: python

                {
                    "ok": True,
                    "data": {
                        "success": True,
                        "key": "insr_live_...",           # the raw key — show once
                        "name": str,
                        "tier": "paid",
                        "dailyLimit": 10000,
                        "creditsAdded": int,
                        "totalCredits": int,
                        "effectiveRate": "$0.04/credit",
                        "chainName": str,
                        "registeredWallet": "0x...",
                        "expiresAt": ISO8601,              # +30 days
                        # Bitcoin only:
                        "btcPaid": float, "btcPrice": float, "usdEquivalent": float,
                        # USDC only:
                        "usdcPaid": float,
                    },
                    "meta": {...},
                }
        """
        body: Dict[str, Any] = {
            "txHash": tx_hash,
            "chainId": chain_id,
            "appName": app_name,
        }
        if amount is not None:
            body["amount"] = amount
        if channel:
            body["channel"] = channel
        # This endpoint is public (no auth) — the transaction sender wallet
        # is the identity.
        response = requests.post(
            f"{self.base_url}/v1/keys/buy",
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def buy_credits(
        self,
        tx_hash: str,
        chain_id: Any,
        amount: Optional[float] = None,
        update_wallet: bool = False,
    ) -> Dict[str, Any]:
        """Top up attestation credits on an existing API key with a USDC or
        BTC payment. Requires the API key the tool spec was initialized with.

        Pre-requisite: the agent or operator must have already broadcast a
        USDC or BTC transfer to the platform wallet BEFORE calling this
        method.

        Args:
            tx_hash: Transaction hash of the USDC or BTC transfer to the
                platform wallet. Must not have been used before.
            chain_id: Either an EVM chain ID (int) for USDC transfers, the
                string ``"solana"`` for USDC on Solana, or the string
                ``"bitcoin"`` for BTC.
            amount: USDC amount paid (required for USDC chains). Not required
                for Bitcoin — USD value is derived from the on-chain BTC
                amount and a price feed.
            update_wallet: If the transaction sender differs from the wallet
                currently registered on this API key, set to ``True`` to
                rebind the registered wallet to the new sender. Defaults to
                ``False``, which rejects mismatched senders with a 403.

        Returns:
            API response envelope:

            .. code-block:: python

                {
                    "ok": True,
                    "data": {
                        "creditsAdded": int,
                        "totalCredits": int,
                        "effectiveRate": "$0.04/credit",
                        "chainName": str,
                        # USDC chains:
                        "usdcPaid": float,
                        # Bitcoin:
                        "btcPaid": float, "btcPrice": float, "usdEquivalent": float,
                    },
                    "meta": {...},
                }
        """
        body: Dict[str, Any] = {
            "txHash": tx_hash,
            "chainId": chain_id,
        }
        if amount is not None:
            body["amount"] = amount
        if update_wallet:
            body["updateWallet"] = True
        return self._post("/v1/credits/buy", body)

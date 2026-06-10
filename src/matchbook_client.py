from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import requests

import config


class MatchbookAPIError(RuntimeError):
    pass


class MatchbookClient:
    def __init__(
        self,
        *,
        username: str | None = None,
        password: str | None = None,
        session: requests.Session | None = None,
        timeout: float | None = None,
    ) -> None:
        self.username = username or config.MATCHBOOK_USERNAME or None
        self.password = password or config.MATCHBOOK_PASSWORD or None
        self.timeout = float(timeout or config.MATCHBOOK_TIMEOUT_SECS)
        self.edge_url = config.MATCHBOOK_EDGE_URL.rstrip("/")
        self.bpapi_url = config.MATCHBOOK_BPAPI_URL.rstrip("/")
        self.session = session or requests.Session()
        self.session.headers.update({
            "accept": "application/json",
            "accept-encoding": "gzip",
            "user-agent": config.USER_AGENT,
        })

    @property
    def session_token(self) -> str | None:
        token = self.session.headers.get("session-token")
        if token:
            return token
        cookie = self.session.cookies.get("session-token")
        return str(cookie) if cookie else None

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_payload: dict[str, Any] | None = None,
        auth_required: bool = False,
    ) -> dict[str, Any]:
        if auth_required and not self.session_token:
            raise MatchbookAPIError("This Matchbook endpoint requires an authenticated session.")

        response = self.session.request(
            method=method,
            url=url,
            params=params,
            json=json_payload,
            timeout=self.timeout,
        )

        if response.status_code >= 400:
            try:
                payload = response.json()
            except ValueError:
                payload = {"message": response.text.strip() or response.reason}
            detail = payload.get("message") or payload.get("error") or payload.get("errors") or response.reason
            raise MatchbookAPIError(f"HTTP {response.status_code}: {detail}")

        try:
            data = response.json()
        except ValueError as exc:
            raise MatchbookAPIError("Matchbook returned a non-JSON response.") from exc

        token = data.get("session-token") or response.cookies.get("session-token")
        if token:
            self.session.headers["session-token"] = str(token)
        return data

    def login(self, username: str | None = None, password: str | None = None) -> dict[str, Any]:
        user = username or self.username
        secret = password or self.password
        if not user or not secret:
            raise MatchbookAPIError("Username and password are required to log in to Matchbook.")
        return self._request(
            "POST",
            f"{self.bpapi_url}/security/session",
            json_payload={"username": user, "password": secret},
        )

    def get_account(self) -> dict[str, Any]:
        return self._request("GET", f"{self.edge_url}/account", auth_required=True)

    def get_balance(self) -> dict[str, Any]:
        return self._request("GET", f"{self.edge_url}/account/balance", auth_required=True)

    def get_events(
        self,
        *,
        sport_ids: int | str | list[int | str] | None = None,
        after: int | None = None,
        before: int | None = None,
        per_page: int = 20,
        include_prices: bool = False,
        price_depth: int = 3,
        include_event_participants: bool = False,
        currency: str | None = None,
    ) -> dict[str, Any]:
        if isinstance(sport_ids, (list, tuple, set)):
            sport_ids_value = ",".join(str(item) for item in sport_ids)
        elif sport_ids is None:
            sport_ids_value = None
        else:
            sport_ids_value = str(sport_ids)

        params: dict[str, Any] = {
            "per-page": int(per_page),
            "include-prices": bool(include_prices),
            "price-depth": int(price_depth),
            "include-event-participants": bool(include_event_participants),
            "exchange-type": "back-lay",
            "odds-type": "DECIMAL",
        }
        if sport_ids_value:
            params["sport-ids"] = sport_ids_value
        if after is not None:
            params["after"] = int(after)
        if before is not None:
            params["before"] = int(before)
        if currency:
            params["currency"] = currency

        return self._request("GET", f"{self.edge_url}/events", params=params)

    def get_horse_racing_events(
        self,
        *,
        hours_ahead: int = 24,
        per_page: int = 10,
        include_prices: bool = False,
        price_depth: int = 3,
        currency: str | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        after = int(now.timestamp())
        before = int((now + timedelta(hours=int(hours_ahead))).timestamp())
        return self.get_events(
            sport_ids=config.MATCHBOOK_HORSE_RACING_SPORT_ID,
            after=after,
            before=before,
            per_page=per_page,
            include_prices=include_prices,
            price_depth=price_depth,
            currency=currency or config.MATCHBOOK_DEFAULT_CURRENCY,
        )

    def get_event_markets(
        self,
        event_id: int | str,
        *,
        include_prices: bool = True,
        price_depth: int = 3,
        per_page: int = 20,
        currency: str | None = None,
        states: str = "open,suspended",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "per-page": int(per_page),
            "include-prices": bool(include_prices),
            "price-depth": int(price_depth),
            "states": states,
            "exchange-type": "back-lay",
            "odds-type": "DECIMAL",
        }
        if currency:
            params["currency"] = currency
        return self._request(
            "GET",
            f"{self.edge_url}/events/{event_id}/markets",
            params=params,
        )
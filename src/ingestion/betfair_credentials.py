"""Windows-account encrypted login settings for unattended historical downloads."""
import base64
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path

import config
from src.ingestion.storage import atomic_json


def connection_path():
    return Path(config.DATA_DIR) / "betfair/auth/connection.json"


def certificate_paths():
    root = connection_path().parent
    return root / "client.crt", root / "client.key"


def _protect(payload, *, decrypt=False):
    if os.name != "nt":
        raise RuntimeError("Saved login settings require Windows account encryption.")

    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]

    source = ctypes.create_string_buffer(payload)
    incoming = Blob(len(payload), ctypes.cast(source, ctypes.POINTER(ctypes.c_ubyte)))
    outgoing = Blob()
    library = ctypes.WinDLL("crypt32", use_last_error=True)
    function = library.CryptUnprotectData if decrypt else library.CryptProtectData
    function.restype = wintypes.BOOL
    function.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    if not function(ctypes.byref(incoming), None, None, None, None, 1, ctypes.byref(outgoing)):
        raise RuntimeError("Windows could not unlock these settings. Reconnect using this Windows account.")
    try:
        return ctypes.string_at(outgoing.data, outgoing.size)
    finally:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.LocalFree.argtypes = [ctypes.c_void_p]
        kernel.LocalFree.restype = ctypes.c_void_p
        kernel.LocalFree(outgoing.data)


def save_connection(username, password, certificate, key):
    value = {"username": username, "password": password, "certificate": str(certificate), "key": str(key)}
    protected = _protect(json.dumps(value).encode("utf-8"))
    atomic_json(connection_path(), {"version": 1, "protected": base64.b64encode(protected).decode("ascii")})


def load_connection():
    path = connection_path()
    if not path.exists():
        return {}
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if envelope["version"] != 1:
            raise ValueError("Unknown format")
        return json.loads(_protect(base64.b64decode(envelope["protected"], validate=True), decrypt=True))
    except (KeyError, ValueError, RuntimeError, OSError):
        raise RuntimeError("Reconnect Betfair in the Data page to unlock the saved login settings.") from None

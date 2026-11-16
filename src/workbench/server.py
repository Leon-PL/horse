"""Loopback-only HTTP API and static client for Racing Workbench."""
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import sqlite3
from urllib.parse import parse_qs, unquote, urlsplit

from src.workbench.data import Catalog
from src.workbench.jobs import Jobs
from src.workbench.review import ReviewStore
from src.workbench.features import Features
from src.workbench.modelling import Modelling
from src.workbench.evaluation import Evaluation
from src.workbench.paper import Paper


class WorkbenchServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, catalog, static_dir):
        if address[0] not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Racing Workbench must bind to loopback")
        self.catalog = catalog
        self.features = Features(catalog)
        self.modelling = Modelling(catalog, self.features)
        self.evaluation = Evaluation(catalog)
        self.paper = Paper()
        self.jobs = Jobs(catalog, modelling=self.modelling)
        self.review = ReviewStore(catalog.root/"workbench/review.sqlite")
        self.static_dir = Path(static_dir)
        self.token = secrets.token_urlsafe(32)
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "RacingWorkbench/1"

    def send(self, value, status=200, content_type="application/json; charset=utf-8", filename=None):
        body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, allow_nan=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def valid_host(self):
        host = self.headers.get("Host", "")
        return host in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

    def do_GET(self):
        self.dispatch(False)

    def do_POST(self):
        self.dispatch(True)

    def dispatch(self, mutation):
        if not self.valid_host():
            return self.send({"error": "Invalid local host"}, 403)
        parsed = urlsplit(self.path)
        parts = [unquote(p) for p in parsed.path.split("/") if p]
        params = {k: v[-1] for k, v in parse_qs(parsed.query).items()}
        if mutation:
            origin = self.headers.get("Origin")
            if self.headers.get("X-Workbench-Token") != self.server.token or (origin and origin != "http://"+self.headers["Host"]):
                return self.send({"error": "Action requires the local app session"}, 403)
            try:
                count = int(self.headers.get("Content-Length", "0"))
                if not 0 < count <= 1000000 or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("Send a JSON action of at most one MB")
                body = json.loads(self.rfile.read(count))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
            except (ValueError, UnicodeDecodeError) as exc:
                return self.send({"error": str(exc)}, 400)
        else:
            body = {}
        try:
            if parts and parts[0] == "api":
                value = self.api(parts[1:], params, body, mutation)
                if value is not None:
                    self.send(value)
                return
            if mutation:
                raise FileNotFoundError("Unknown action")
            name = {"/": "index.html", "/app.js": "app.js", "/features.js": "features.js",
                    "/modelling.js": "modelling.js", "/evaluation.js": "evaluation.js", "/paper.js": "paper.js",
                    "/style.css": "style.css"}.get(parsed.path)
            if not name:
                raise FileNotFoundError("Page was not found")
            kind = {"html": "text/html", "js": "text/javascript", "css": "text/css"}[name.rsplit(".", 1)[1]]
            self.send((self.server.static_dir/name).read_bytes(), content_type=f"{kind}; charset=utf-8")
        except FileNotFoundError as exc:
            self.send({"error": str(exc)}, 404)
        except (ValueError, KeyError, TypeError) as exc:
            self.send({"error": str(exc)}, 400)
        except (OSError, sqlite3.Error) as exc:
            self.send({"error": "Source is temporarily unavailable: "+str(exc)}, 503)
        except Exception:
            import logging
            logging.exception("Workbench request failed")
            self.send({"error": "The request failed; inspect the local server log"}, 500)

    def api(self, parts, params, body, mutation):
        catalog = self.server.catalog
        dataset = params.get("dataset") or None
        yesterday = (date.today()-timedelta(days=1)).isoformat()
        end = params.get("end") or catalog.latest_day() or yesterday
        start = params.get("start") or (date.fromisoformat(end)-timedelta(days=6)).isoformat()
        route = "/".join(parts)
        if mutation:
            if route == "jobs":
                return self.server.jobs.start(body.get("kind"), body.get("params", {}))
            if route == "workers/atr/stop":
                return self.server.jobs.stop_atr()
            if route == "review/note":
                catalog.canonical(body["race_id"])
                return self.server.review.note(body["race_id"], body.get("note", ""), body.get("bookmarked", False))
            if route == "review/decision":
                return self.server.review.decide(body["issue_id"], body["state"], body.get("note", ""))
            if route == "review/proposals":
                return self.server.review.propose(catalog, body["race_id"], body["changes"], body.get("reason", ""), body.get("evidence", ""), body["fingerprint"])
            raise FileNotFoundError("Action was not found")
        if route == "session":
            return {"token": self.server.token, "latest_day": catalog.latest_day(), "sources": list(__import__("src.workbench.data", fromlist=["SOURCES"]).SOURCES)}
        if route == "overview":
            return catalog.overview(dataset)
        if route == "datasets":
            return {"rows": catalog.manifests()}
        if route == "paper":
            return self.server.paper.overview()
        if route == "paper/day":
            return self.server.paper.day(params.get("date") or None)
        if route == "features/overview":
            return self.server.features.overview(dataset)
        if parts[:1] == ["modelling"]:
            modelling = self.server.modelling
            if route == "modelling/overview":
                return modelling.overview(dataset)
            if route == "modelling/benchmarks":
                return modelling.benchmarks(dataset)
            if route == "modelling/runs":
                return modelling.runs(dataset)
            if route == "modelling/evaluation":
                return catalog.cached("evaluation_menu", 10, self.server.evaluation.menu)
            if route == "modelling/evaluation/price-timing":
                return self.server.evaluation.timing(params)
            if len(parts) == 3 and parts[1] == "evaluation":
                return self.server.evaluation.run(parts[2], params)
            if len(parts) == 3 and parts[1] == "benchmarks":
                return modelling.benchmark(parts[2], dataset)[0]
            if len(parts) == 3 and parts[1] == "runs":
                return modelling.run(parts[2], dataset)
            if len(parts) == 4 and parts[1] == "runs" and parts[3] == "importance":
                return modelling.importance(parts[2])
            raise FileNotFoundError("Modelling view was not found")
        if route == "features/catalog":
            return self.server.features.catalog_rows(params.get("feature_set"), params.get("query", ""),
                params.get("family", ""), params.get("offset", 0), params.get("limit", 60))
        if route == "features/quality":
            return self.server.features.quality(start, end, params.get("feature_set"), params.get("family", ""))
        if route == "features/races":
            return self.server.features.races(start, end)
        if len(parts) >= 3 and parts[:2] == ["features", "races"]:
            if len(parts) == 4 and parts[3] == "value":
                return self.server.features.value(parts[2], params["horse_id"], params["column"])
            if len(parts) not in {3, 4} or len(parts) == 4 and parts[3] != "export":
                raise FileNotFoundError("Feature route was not found")
            result = self.server.features.race(parts[2], params.get("feature_set"), params.get("family", ""), params.get("query", ""), params.get("offset", 0))
            if len(parts) == 4 and parts[3] == "export":
                import pandas as pd
                frame = pd.DataFrame(result["rows"])
                frame["feature_generation"] = result["store"]["signature"]
                frame["source_dataset"] = result["store"]["dataset"]
                frame["feature_build"] = result["store"]["build_id"]
                frame["feature_set"] = result["feature_set"]
                frame["feature_set_signature"] = result["feature_set_signature"]
                self.send(frame.to_csv(index=False).encode(), content_type="text/csv; charset=utf-8", filename="race-features.csv")
                return None
            if len(parts) != 3:
                raise FileNotFoundError("Feature route was not found")
            return result
        if route == "races":
            return catalog.races(start, end, params.get("query", ""), params.get("track", ""), params.get("limit", 50), params.get("offset", 0))
        if len(parts) >= 2 and parts[0] == "races":
            identifier = parts[1]
            if len(parts) == 2:
                return catalog.race(identifier, dataset)
            if parts[2] == "source":
                return catalog.race_source(identifier, params["source"], params.get("layer", "accepted"), dataset)
            if parts[2] == "prepared":
                return catalog.prepared(identifier, dataset)
            if parts[2] == "compare":
                return catalog.compare(identifier, params["source"], params["before"], params["after"])
            if parts[2] == "export":
                race = catalog.race(identifier, dataset)
                if params.get("format") == "csv":
                    import pandas as pd
                    self.send(pd.DataFrame(race["rows"]).to_csv(index=False).encode(), content_type="text/csv; charset=utf-8", filename=f"race-{identifier}.csv")
                else:
                    from src.workbench.data import SOURCES
                    pack = {"race": race, "sources": {}}
                    for provider in SOURCES:
                        for layer in ("accepted", "native"):
                            try:
                                pack["sources"][provider+"_"+layer] = catalog.race_source(identifier, provider, layer, dataset)
                            except (OSError, ValueError, sqlite3.Error) as exc:
                                pack["sources"][provider+"_"+layer] = {"error": str(exc)}
                    self.send(pack, filename=f"race-{identifier}-evidence.json")
                return None
        if route == "coverage":
            return catalog.coverage(start, end, dataset)
        if route == "bha/fixtures":
            return catalog.bha_fixtures(start, end, params.get("layer", "native"), dataset)
        if route == "workers/atr/log":
            from src.workbench.data import read_json
            launch = read_json(catalog.root/"atr_cache/rebuild_v2/launch_options.json")
            if launch.get("job_id"):
                return self.server.jobs.log(launch["job_id"])
            path = Path(launch.get("stderr", catalog.root/"atr_cache/rebuild_v2/worker.stderr.log")).resolve()
            if not path.is_relative_to(catalog.root):
                raise ValueError("Worker log is outside the data directory")
            if not path.exists():
                return {"text": "Worker log is not available"}
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size-64000))
                return {"text": stream.read().decode("utf-8", errors="replace")}
        if route == "issues":
            return catalog.issues(start, end)
        if route == "jobs":
            return {"rows": self.server.jobs.list(), "workers": catalog.workers()}
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "log":
            return self.server.jobs.log(parts[1])
        if route == "review":
            return self.server.review.read()
        if len(parts) == 3 and parts[:2] == ["review", "proposals"]:
            self.send(self.server.review.proposal(parts[2]), filename=f"correction-{parts[2]}.json")
            return None
        if route == "evidence":
            raw, content_type = catalog.evidence(params["source"], params["sha"])
            self.send(raw, content_type=content_type, filename=params["sha"]+(".pdf" if content_type == "application/pdf" else ".txt"))
            return None
        raise FileNotFoundError("API route was not found")


def main():
    import argparse
    import config
    parser = argparse.ArgumentParser(description="Local racing sources, feature inspection and project jobs")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = WorkbenchServer(("127.0.0.1", args.port), Catalog(config.DATA_DIR, config.BASE_DIR), Path(config.BASE_DIR)/"ui/dist")
    from src.ingestion.storage import atomic_json
    from datetime import datetime, timezone
    import os
    atomic_json(Path(config.DATA_DIR)/"workbench/server_status.json", {"pid": os.getpid(),
                "url": f"http://127.0.0.1:{server.server_port}", "started_at": datetime.now(timezone.utc).isoformat()})
    print(f"Racing Workbench: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

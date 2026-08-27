from preflight import check
from train import train
from evaluate import evaluate


if __name__ == "__main__":
    status = check()
    print(status)
    if not status["ok"]:
        raise SystemExit(
            status.get("error") or status.get("disk_warning")
            or status.get("next") or "Alternative preflight failed"
        )
    print(train())
    print(evaluate())

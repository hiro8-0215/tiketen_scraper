from preflight import check
from train_policy import train
from evaluate import evaluate


if __name__ == "__main__":
    status = check()
    print(status)
    if not status["ok"]:
        raise SystemExit(
            status.get("error") or status.get("next")
            or "Prepare OOF inputs first"
        )
    print(train())
    print(evaluate())

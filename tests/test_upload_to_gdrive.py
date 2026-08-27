import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import upload_to_gdrive


def successful_response():
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps({
        "status": "success",
        "fileId": "file-id",
    }).encode("utf-8")
    return response


class DriveUploadRetryTest(unittest.TestCase):
    def test_transient_404_is_retried(self):
        error = urllib.error.HTTPError(
            "https://example.invalid/exec", 404, "Not Found", {}, None
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "ae-group_master.csv")
            path.write_text("ticket_id\n", encoding="utf-8")
            with (
                patch.object(
                    upload_to_gdrive.urllib.request,
                    "urlopen",
                    side_effect=[error, successful_response()],
                ) as urlopen,
                patch.object(upload_to_gdrive.time, "sleep") as sleep,
                patch("builtins.print"),
            ):
                upload_to_gdrive.upload_file(
                    "https://example.invalid/exec", "token", path, "data_8_27"
                )

        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_access_denial_is_not_retried(self):
        error = urllib.error.HTTPError(
            "https://example.invalid/exec", 403, "Forbidden", {}, None
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "group_master.csv")
            path.write_text("ticket_id\n", encoding="utf-8")
            with (
                patch.object(
                    upload_to_gdrive.urllib.request,
                    "urlopen",
                    side_effect=error,
                ) as urlopen,
                patch.object(upload_to_gdrive.time, "sleep") as sleep,
                self.assertRaises(RuntimeError),
            ):
                upload_to_gdrive.upload_file(
                    "https://example.invalid/exec", "token", path, "data_8_27"
                )

        urlopen.assert_called_once()
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()

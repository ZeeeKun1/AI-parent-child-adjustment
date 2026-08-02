from coregulation_poc.connection_check import check_realtime_connection
from coregulation_poc.settings import Settings


def test_connection_check_stops_before_network_without_key() -> None:
    settings = Settings(dashscope_api_key=None, aliyun_workspace_id="ws-test")

    result = check_realtime_connection(settings)

    assert result == {"ok": False, "stage": "configuration", "error": "missing_api_key"}

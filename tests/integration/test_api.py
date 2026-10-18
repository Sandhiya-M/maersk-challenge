from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_shipments_200_and_404() -> None:
    response = client.get("/shipments")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

    missing = client.get("/shipments/MISSING")
    assert missing.status_code == 404


def test_dq_report_endpoint() -> None:
    response = client.get("/data-quality")
    assert response.status_code == 200
    assert "status" in response.json()


def test_shipments_pagination() -> None:
    response = client.get("/shipments?offset=1&limit=2")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) <= 2
    assert payload[0]["shipment_id"]


def test_shipments_large_page_limit() -> None:
    response = client.get("/shipments?offset=0&limit=5000")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) <= 5000


def test_port_events_pagination() -> None:
    response = client.get("/port-events?offset=0&limit=2")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) <= 2
    assert payload[0]["event_id"]


def test_ports_endpoint() -> None:
    response = client.get("/ports")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert payload
    assert "port_code" in payload[0]


def test_assistant_chat_endpoint() -> None:
    response = client.post("/assistant/chat", json={"query": "Which route has the highest average delay this quarter?"})
    assert response.status_code == 200
    body = response.json()
    assert "response" in body
    assert isinstance(body["response"], str)
    assert body["response"]


def test_assistant_route_stats_query() -> None:
    response = client.post("/assistant/chat", json={"query": "Show me the route statistics for Singapore to Los Angeles."})
    assert response.status_code == 200
    body = response.json()
    assert "response" in body
    assert "I can only answer" not in body["response"]
    text = body["response"].lower()
    assert (
        "route" in text
        or "credit" in text
        or "quota" in text
        or "billing" in text
    )


def test_assistant_uses_ollama_only(monkeypatch) -> None:
    from src.assistant import llm_client

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    client = llm_client.LLMClient(model="qwen2.5:3b")

    assert client.provider == "ollama"
    assert client.api_key == "ollama"
    assert client.model == "qwen2.5:3b"


def test_assistant_falls_back_when_model_ignores_tools(monkeypatch) -> None:
    from src.assistant import llm_client
    from src.assistant.tools import TOOL_DEFINITIONS

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    responses = iter([
        {"message": {"content": "I don't have direct access to historical delivery data.", "tool_calls": None}},
        {"message": {"content": "Top delay routes this quarter: KRPUS->CNSZX (166.0h avg, 1 shipments). [source: get_top_delay_routes]"}},
    ])

    def fake_post(url, json, timeout):
        assert url.endswith("/api/chat")
        return FakeResponse(next(responses))

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm_client.httpx, "post", fake_post)

    result = llm_client.LLMClient(model="qwen2.5:3b").generate_with_tools(
        "which route was delivered with the highest time in 2024",
        tools=TOOL_DEFINITIONS,
        max_calls=1,
    )

    assert result["status"] == "ok"
    assert result["tool_calls"]
    assert "direct access" not in result["final_output"].lower()


def test_assistant_uses_ollama_api_for_tools(monkeypatch) -> None:
    from src.assistant import llm_client
    from src.assistant.tools import TOOL_DEFINITIONS

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    responses = iter([
        {"message": {"content": "", "tool_calls": [{"function": {"name": "get_top_delay_routes", "arguments": {"period": "quarter", "limit": 1}}}]}},
        {"message": {"content": "Top delay routes this quarter: KRPUS->CNSZX (166.0h avg, 1 shipments). [source: get_top_delay_routes]"}},
    ])

    def fake_post(url, json, timeout):
        assert url.endswith("/api/chat")
        assert json["model"] == "qwen2.5:3b"
        assert json["tools"]
        return FakeResponse(next(responses))

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setattr(llm_client.httpx, "post", fake_post)

    result = llm_client.LLMClient(model="qwen2.5:3b").generate_with_tools(
        "Which route had the highest average delay this quarter?",
        tools=TOOL_DEFINITIONS,
        max_calls=2,
    )

    assert result["status"] == "ok"
    assert result["tool_calls"]
    assert "get_top_delay_routes" in result["final_output"] or "KRPUS->CNSZX" in result["final_output"]


def test_prediction_endpoint() -> None:
    payload = {
        "shipment_id": "SHP-001",
        "booking_time": "2024-01-15T12:00:00",
        "port_of_loading": "SGSIN",
        "port_of_discharge": "USLAX",
        "cargo_type": "Electronics",
        "container_count": 12,
        "weight_tons": 1450,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "is_more_than_24h_late" in body
    assert "probability_gt_24h_late" in body
    assert "predicted_delay_hours" in body


def test_prediction_delay_endpoint() -> None:
    payload = {
        "shipment_id": "SHP-002",
        "booking_time": "2024-02-20T08:00:00",
        "port_of_loading": "SGSIN",
        "port_of_discharge": "USLAX",
        "cargo_type": "Food & Beverage",
        "container_count": 18,
        "weight_tons": 2200,
    }
    response = client.post("/predict-delay", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["shipment_id"] == "SHP-002"
    assert "is_more_than_24h_late" in body
    assert 0.0 <= body["probability_gt_24h_late"] <= 1.0


def test_model_metrics_endpoint() -> None:
    response = client.get("/predict/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert "f1" in payload or "precision" in payload
    assert "threshold" in payload or "auc" in payload


def test_pipeline_run_endpoint() -> None:
    response = client.post("/pipeline/run")
    assert response.status_code == 200
    payload = response.json()
    assert "pipeline" in payload
    assert "raw_quality" in payload


def test_pipeline_upload_endpoint() -> None:
    csv_content = "shipment_id,origin_port,destination_port,booking_date,status\nSHP-UP-001,SGSIN,USLAX,2024-01-20,DELAYED\n"
    response = client.post(
        "/pipeline/upload",
        files={"shipments": ("shipments.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["saved"]["shipments"] == "shipments.csv"


def test_route_stats_endpoint() -> None:
    response = client.get("/routes/SGSIN/USLAX/stats")
    assert response.status_code == 200
    payload = response.json()
    assert "origin" in payload
    assert "destination" in payload
    assert "average_delay_hours" in payload
    assert "on_time_rate" in payload
    assert "shipment_count" in payload
    assert "shipments" in payload or "scheduled_shipments" in payload

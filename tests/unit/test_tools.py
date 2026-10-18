from src.assistant.tools import get_route_stats, get_top_delay_routes, query_shipments


def test_query_shipments_returns_data() -> None:
    rows = query_shipments()
    assert len(rows) >= 1
    assert "shipment_id" in rows[0]


def test_query_shipments_filters_by_route() -> None:
    rows = query_shipments({"origin_port": "SGSIN", "destination_port": "USLAX"})
    assert rows
    assert all(row.get("origin_port") == "SGSIN" for row in rows)
    assert all(row.get("destination_port") == "USLAX" for row in rows)


def test_get_route_stats_has_expected_fields() -> None:
    stats = get_route_stats("SGSIN", "USLAX")
    assert "average_delay_hours" in stats
    assert "on_time_rate" in stats
    assert "shipment_count" in stats
    assert stats["origin"].upper() == "SGSIN"
    assert stats["destination"].upper() == "USLAX"


def test_get_top_delay_routes_returns_routed_results() -> None:
    routes = get_top_delay_routes(period="quarter", limit=3)
    assert routes
    assert all("route" in row for row in routes)
    assert all("average_delay_hours" in row for row in routes)
    assert routes[0]["average_delay_hours"] >= routes[-1]["average_delay_hours"]

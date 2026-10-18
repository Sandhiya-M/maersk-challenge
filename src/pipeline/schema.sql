CREATE TABLE IF NOT EXISTS shipments (
    shipment_id VARCHAR,
    booking_time TIMESTAMP,
    port_of_loading VARCHAR,
    port_of_discharge VARCHAR,
    vessel_name VARCHAR,
    cargo_type VARCHAR,
    delay_hours DOUBLE,
    shipment_status VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ports (
    port_code VARCHAR,
    port_name VARCHAR,
    country VARCHAR,
    region VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS port_events (
    event_id VARCHAR,
    port_code VARCHAR,
    event_time TIMESTAMP,
    event_type VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

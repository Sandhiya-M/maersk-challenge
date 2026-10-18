# Data Quality Decisions

- Fail fast when required schema columns are missing.
- Reject rows with null values in key business identifiers.
- Treat duplicate shipment IDs as a critical integrity issue.
- Keep the quality checker standalone and free of database dependencies.

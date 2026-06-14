# Metrics Architecture

## System Design
The metrics system will be an extension of the existing logging infrastructure, transforming the `logs` table into a source of truth for observability.

## Data Layer Changes
### Database Schema Update
The `logs` table in `proxy.db` will be extended with three new columns:
- `tokens_sent` (INTEGER): Prompt token count.
- `tokens_received` (INTEGER): Completion token count.
- `latency_ms` (INTEGER): Total request duration in milliseconds.

### Database Operations
- `database.add_log()` will be updated to accept these new parameters.
- A new function `database.get_metrics_summary()` will be added to perform SQL aggregations:
  - `COUNT(*)` grouped by `provider_name` for request volume.
  - `SUM(tokens_sent)` and `SUM(tokens_received)` grouped by `provider_name` for token usage.
  - `AVG(latency_ms)` grouped by `provider_name` for performance.

## Logic Layer Changes
### Latency Measurement
In `core/router.py`, the `RouterService` will be modified to:
1. Capture a start timestamp before the HTTP call.
2. Capture an end timestamp after the response is received.
3. Calculate `latency_ms = (end - start) * 1000`.

### Token Extraction
The `RouterService` will extract usage data from the provider's response body:
- **OpenAI format**: Extract from `usage.prompt_tokens` and `usage.completion_tokens`.
- **Anthropic format**: Extract from `usage.input_tokens` and `usage.output_tokens`.
- This extraction will occur in both `_handle_non_streaming` and `_handle_streaming` (where usage is typically sent in the final chunk).

## API Layer Changes
A new endpoint `GET /api/metrics` will be created:
- **Input**: Optional time range (e.g., `?days=7`).
- **Output**: A JSON object containing aggregated statistics per provider.

## Frontend Layer Changes
### Metrics Page
- Integrate a lightweight charting library (e.g., **Chart.js**) into `static/app.js`.
- Create a new navigation tab for "Metrics".
- Implement three charts:
  - Tokens Sent vs Received (Stacked Bar Chart).
  - Request Count (Pie Chart).
  - Average Latency (Bar Chart).

### Provider Dashboard
- Update the provider list rendering logic to call the metrics API and display the average latency next to each provider's name.

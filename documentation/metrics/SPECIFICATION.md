# Metrics Specification

## Overview
The goal is to add observability to the Route.LLM dashboard, allowing users to monitor usage patterns, cost (via tokens), and performance (via latency) across different LLM providers.

## Requirements

### 1. Metrics Tracking (Completed)
The system tracks the following for every request:
- **Latency**: Captured in `core/router.py` (TTFB for streaming, total for non-streaming).
- **Token Usage**: Extracted from responses in `core/router.py`.

### 2. Dashboard Visualizations (Completed)
Added a new "Metrics" view with:
- **Total Tokens Graph**: Stacked bar chart in `static/app.js`.
- **Request Volume Graph**: Pie chart in `static/app.js`.
- **Latency Chart**: Bar chart showing average latency.

### 3. Provider Dashboard Integration (Completed)
- Average latency displayed on provider cards in the dashboard.

## Data Definitions
- **Latency**: Measured in milliseconds (ms).
- **Tokens Sent**: Integer count of prompt tokens.
- **Tokens Received**: Integer count of completion tokens.
- **Provider**: The name of the configured provider.

## Technology Stack
- **Charting Library**: Chart.js (chosen for its lightweight nature and ease of integration with vanilla JS).

## Constraints
- Metrics should be derived from the existing `logs` table to avoid adding complex new storage systems.
- Aggregations should be performed on the server side to keep the frontend lightweight.

# Metrics Specification

## Overview
The goal is to add observability to the Route.LLM dashboard, allowing users to monitor usage patterns, cost (via tokens), and performance (via latency) across different LLM providers.

## Requirements

### 1. Metrics Tracking
The system must track the following for every request:
- **Latency**: The time taken from sending the request to receiving the full response (or the first byte for streaming).
- **Token Usage**: Number of prompt tokens (sent) and completion tokens (received).

### 2. Dashboard Visualizations
A new "Metrics" view should be added to the dashboard containing:
- **Total Tokens Graph**: A chart showing the distribution of tokens sent and received per provider.
- **Request Volume Graph**: A chart showing the number of requests routed to each provider.
- **Latency Heatmap/Bar Chart**: A chart showing the average latency per provider.

### 3. Provider Dashboard Integration
- The provider list/modal should display the **Average Latency** for that specific provider based on recent history.

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

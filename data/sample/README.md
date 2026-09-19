# Development Flight Sample

This directory contains a small, structured development sample (`bts_2024_dev_sample.csv`) used strictly for verifying pipeline execution, CI/CD, and automated integration testing.

## Attribution & Scope
* **Purpose**: Local pipeline smoke testing and CLI demonstration.
* **Schema**: Follows the U.S. DOT Bureau of Transportation Statistics (BTS) TranStats Reporting Carrier On-Time Performance schema.
* **Sample Size**: 500 flights across major domestic carriers and airports.
* **Note**: This is a development sample. For full-scale production model training in Phase 2, download full historical month/year extracts from the official BTS TranStats portal and place them into `data/raw/` as detailed in `data/raw/README.md`.

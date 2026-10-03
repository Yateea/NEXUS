-- ==============================================================
-- NEXUS - Migration : ajout de departure_timestamp sur fact_transport (ML-04)
-- A executer une seule fois sur une base deja initialisee :
--   docker exec -i nexus_postgres psql -U nexus_user -d nexus_db < 03_add_transport_departure_time.sql
-- ==============================================================

ALTER TABLE fact_transport ADD COLUMN IF NOT EXISTS departure_timestamp TIMESTAMP;

CREATE INDEX IF NOT EXISTS idx_fact_transport_departure ON fact_transport(departure_timestamp);

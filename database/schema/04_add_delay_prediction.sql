-- ==============================================================
-- NEXUS - Migration : ajout de fact_delay_prediction (ML-04 Delay Prediction)
-- A executer une seule fois sur une base deja initialisee :
--   docker exec -i nexus_postgres psql -U nexus_user -d nexus_db < 04_add_delay_prediction.sql
-- ==============================================================

CREATE TABLE IF NOT EXISTS fact_delay_prediction (
    prediction_id            BIGSERIAL PRIMARY KEY,
    transport_id             BIGINT NOT NULL REFERENCES fact_transport(transport_id),
    delay_probability        NUMERIC(5,4),   -- 0-1, probabilite de retard (>15 min)
    expected_delay_minutes   NUMERIC(10,2),  -- retard attendu en minutes
    actual_delay_minutes     NUMERIC(10,2),  -- retard reellement observe (pour comparaison/validation)
    model_name               VARCHAR(60)
);

CREATE INDEX IF NOT EXISTS idx_fact_delay_prediction_transport ON fact_delay_prediction(transport_id);

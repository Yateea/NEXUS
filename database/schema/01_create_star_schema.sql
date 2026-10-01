-- ==============================================================
-- NEXUS - Star Schema complet (section 15 du cahier des charges)
-- Ce fichier est execute automatiquement par Postgres au premier
-- demarrage du conteneur (docker-entrypoint-initdb.d), s'il est
-- place dans database/schema/ AVANT le tout premier `docker compose up`.
-- ==============================================================

-- ==========================================================
-- DIMENSIONS
-- ==========================================================

CREATE TABLE dim_date (
    date_id         INT PRIMARY KEY,          -- format YYYYMMDD
    date            DATE NOT NULL UNIQUE,
    day             INT NOT NULL,
    day_of_week     INT NOT NULL,             -- 1=lundi ... 7=dimanche
    week            INT NOT NULL,
    month           INT NOT NULL,
    quarter         INT NOT NULL,
    year            INT NOT NULL,
    is_weekend      BOOLEAN NOT NULL DEFAULT FALSE,
    is_holiday      BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE dim_site (
    site_id                 VARCHAR(20) PRIMARY KEY,
    site_name               VARCHAR(100) NOT NULL,
    country                 VARCHAR(60) NOT NULL,
    city                    VARCHAR(60) NOT NULL,
    latitude                NUMERIC(9,6),
    longitude               NUMERIC(9,6),
    production_capacity     NUMERIC(14,2),
    warehouse_capacity      NUMERIC(14,2)
);

CREATE TABLE dim_warehouse (
    warehouse_id     VARCHAR(20) PRIMARY KEY,
    site_id          VARCHAR(20) NOT NULL REFERENCES dim_site(site_id),
    warehouse_type   VARCHAR(50),
    capacity_units   NUMERIC(14,2),
    latitude         NUMERIC(9,6),
    longitude        NUMERIC(9,6)
);

CREATE TABLE dim_production_line (
    line_id             VARCHAR(20) PRIMARY KEY,
    site_id             VARCHAR(20) NOT NULL REFERENCES dim_site(site_id),
    line_name           VARCHAR(100),
    production_type     VARCHAR(60),
    capacity_per_hour   NUMERIC(12,2)
);

CREATE TABLE dim_machine (
    machine_id              VARCHAR(20) PRIMARY KEY,
    site_id                 VARCHAR(20) NOT NULL REFERENCES dim_site(site_id),
    line_id                 VARCHAR(20) REFERENCES dim_production_line(line_id),
    machine_type            VARCHAR(60),
    installation_date       DATE,
    manufacturer            VARCHAR(100),
    nominal_capacity        NUMERIC(12,2),
    maintenance_interval    INT  -- en heures ou jours, a preciser selon convention retenue
);

CREATE TABLE dim_vehicle (
    vehicle_id              VARCHAR(20) PRIMARY KEY,
    vehicle_type            VARCHAR(50),
    capacity_kg             NUMERIC(10,2),
    fuel_type               VARCHAR(30),
    consumption_per_100km   NUMERIC(6,2),
    home_site_id            VARCHAR(20) REFERENCES dim_site(site_id)
);

CREATE TABLE dim_supplier (
    supplier_id          VARCHAR(20) PRIMARY KEY,
    supplier_name        VARCHAR(150) NOT NULL,
    country              VARCHAR(60),
    city                 VARCHAR(60),
    lead_time_days       INT,
    payment_terms        VARCHAR(60),
    reliability_score    NUMERIC(5,2)  -- 0-100
);

CREATE TABLE dim_product (
    product_id                 VARCHAR(20) PRIMARY KEY,
    product_name               VARCHAR(150) NOT NULL,
    category_id                VARCHAR(20),
    subcategory_id             VARCHAR(20),
    unit_cost                  NUMERIC(12,2),
    selling_price              NUMERIC(12,2),
    weight                     NUMERIC(10,3),
    volume                     NUMERIC(10,3),
    safety_stock               NUMERIC(12,2),
    reorder_point              NUMERIC(12,2),
    lead_time_days             INT,
    production_time_minutes    NUMERIC(10,2)
);

CREATE TABLE dim_customer (
    customer_id          VARCHAR(20) PRIMARY KEY,
    customer_segment     VARCHAR(60),
    country              VARCHAR(60),
    city                 VARCHAR(60),
    region               VARCHAR(60)
);

-- Optionnel - V2 (mentionne en section 15)
CREATE TABLE dim_employee (
    employee_id     VARCHAR(20) PRIMARY KEY,
    site_id         VARCHAR(20) REFERENCES dim_site(site_id),
    department      VARCHAR(60),
    role            VARCHAR(60),
    shift           VARCHAR(30)
);

-- ==========================================================
-- FACTS
-- ==========================================================

CREATE TABLE fact_sales (
    sales_id      BIGSERIAL PRIMARY KEY,
    date_id       INT NOT NULL REFERENCES dim_date(date_id),
    product_id    VARCHAR(20) NOT NULL REFERENCES dim_product(product_id),
    customer_id   VARCHAR(20) NOT NULL REFERENCES dim_customer(customer_id),
    site_id       VARCHAR(20) REFERENCES dim_site(site_id),
    quantity      NUMERIC(12,2) NOT NULL CHECK (quantity >= 0),
    unit_price    NUMERIC(12,2) NOT NULL CHECK (unit_price >= 0),
    discount      NUMERIC(12,2) DEFAULT 0 CHECK (discount >= 0),
    revenue       NUMERIC(14,2) NOT NULL
);

CREATE TABLE fact_inventory (
    inventory_id         BIGSERIAL PRIMARY KEY,
    date_id              INT NOT NULL REFERENCES dim_date(date_id),
    product_id           VARCHAR(20) NOT NULL REFERENCES dim_product(product_id),
    warehouse_id         VARCHAR(20) NOT NULL REFERENCES dim_warehouse(warehouse_id),
    opening_stock        NUMERIC(14,2) NOT NULL,
    inbound_quantity     NUMERIC(14,2) NOT NULL DEFAULT 0,
    production_quantity  NUMERIC(14,2) NOT NULL DEFAULT 0,
    sales_quantity       NUMERIC(14,2) NOT NULL DEFAULT 0,
    outbound_quantity    NUMERIC(14,2) NOT NULL DEFAULT 0,
    -- closing_stock = opening_stock + inbound + production - sales - outbound (regle metier section 15)
    closing_stock        NUMERIC(14,2) NOT NULL,
    stock_value          NUMERIC(14,2)
);

CREATE TABLE fact_purchase (
    purchase_id       BIGSERIAL PRIMARY KEY,
    date_id           INT NOT NULL REFERENCES dim_date(date_id),
    supplier_id       VARCHAR(20) NOT NULL REFERENCES dim_supplier(supplier_id),
    product_id        VARCHAR(20) NOT NULL REFERENCES dim_product(product_id),
    order_date_id     INT REFERENCES dim_date(date_id),
    expected_date_id  INT REFERENCES dim_date(date_id),
    actual_date_id    INT REFERENCES dim_date(date_id),
    quantity          NUMERIC(14,2) NOT NULL,
    unit_price        NUMERIC(12,2) NOT NULL,
    total_cost        NUMERIC(14,2),
    is_delayed        BOOLEAN DEFAULT FALSE,
    delay_days        INT DEFAULT 0
);

CREATE TABLE fact_production (
    production_id             BIGSERIAL PRIMARY KEY,
    date_id                   INT NOT NULL REFERENCES dim_date(date_id),
    site_id                   VARCHAR(20) NOT NULL REFERENCES dim_site(site_id),
    line_id                   VARCHAR(20) NOT NULL REFERENCES dim_production_line(line_id),
    machine_id                VARCHAR(20) REFERENCES dim_machine(machine_id),
    product_id                VARCHAR(20) NOT NULL REFERENCES dim_product(product_id),
    planned_quantity          NUMERIC(14,2) NOT NULL,
    actual_quantity           NUMERIC(14,2) NOT NULL,
    defect_quantity           NUMERIC(14,2) NOT NULL DEFAULT 0,
    production_time_minutes   NUMERIC(12,2),
    downtime_minutes          NUMERIC(12,2) DEFAULT 0
);

CREATE TABLE fact_energy (
    energy_id     BIGSERIAL PRIMARY KEY,
    timestamp     TIMESTAMP NOT NULL,
    site_id       VARCHAR(20) NOT NULL REFERENCES dim_site(site_id),
    line_id       VARCHAR(20) REFERENCES dim_production_line(line_id),
    machine_id    VARCHAR(20) REFERENCES dim_machine(machine_id),
    energy_kwh    NUMERIC(14,4) NOT NULL,
    power_kw      NUMERIC(12,4),
    temperature   NUMERIC(6,2),
    humidity      NUMERIC(6,2)
);

CREATE TABLE fact_maintenance (
    maintenance_id     BIGSERIAL PRIMARY KEY,
    machine_id         VARCHAR(20) NOT NULL REFERENCES dim_machine(machine_id),
    date_id            INT NOT NULL REFERENCES dim_date(date_id),
    maintenance_type   VARCHAR(50),   -- preventive / corrective
    duration_hours     NUMERIC(8,2),
    cost               NUMERIC(12,2),
    failure_flag       BOOLEAN DEFAULT FALSE
);

CREATE TABLE fact_transport (
    transport_id        BIGSERIAL PRIMARY KEY,
    date_id             INT NOT NULL REFERENCES dim_date(date_id),
    vehicle_id          VARCHAR(20) NOT NULL REFERENCES dim_vehicle(vehicle_id),
    warehouse_id        VARCHAR(20) REFERENCES dim_warehouse(warehouse_id),
    destination          VARCHAR(150),
    distance_km          NUMERIC(10,2),
    planned_duration      NUMERIC(10,2),  -- minutes
    actual_duration       NUMERIC(10,2),  -- minutes
    fuel_consumption      NUMERIC(10,2),
    delivery_status       VARCHAR(30),    -- on_time / delayed / failed
    delay_minutes         NUMERIC(10,2) DEFAULT 0
);

CREATE TABLE fact_machine_sensor (
    sensor_reading_id   BIGSERIAL PRIMARY KEY,
    timestamp           TIMESTAMP NOT NULL,
    machine_id          VARCHAR(20) NOT NULL REFERENCES dim_machine(machine_id),
    temperature         NUMERIC(8,3),
    vibration           NUMERIC(8,3),
    pressure            NUMERIC(8,3),
    rpm                 NUMERIC(10,2),
    power               NUMERIC(10,3),
    operating_hours     NUMERIC(12,2)
);

CREATE TABLE fact_demand_forecast (
    forecast_id        BIGSERIAL PRIMARY KEY,
    date_id            INT NOT NULL REFERENCES dim_date(date_id),  -- date de reference du forecast
    product_id         VARCHAR(20) NOT NULL REFERENCES dim_product(product_id),
    forecast_date      DATE NOT NULL,      -- date cible predite
    forecast_quantity  NUMERIC(14,2) NOT NULL,
    lower_bound        NUMERIC(14,2),
    upper_bound        NUMERIC(14,2),
    model_name         VARCHAR(60)
);

CREATE TABLE fact_anomaly (
    anomaly_id       BIGSERIAL PRIMARY KEY,
    timestamp        TIMESTAMP NOT NULL,
    domain           VARCHAR(50) NOT NULL,   -- supply_chain / inventory / production / energy / maintenance / transport
    entity_id        VARCHAR(50) NOT NULL,   -- id du produit/machine/vehicule concerne
    metric           VARCHAR(80) NOT NULL,
    actual_value     NUMERIC(14,4),
    expected_value   NUMERIC(14,4),
    anomaly_score    NUMERIC(6,4),
    severity         VARCHAR(20),            -- LOW / MEDIUM / HIGH / CRITICAL
    model            VARCHAR(60)
);

-- ==========================================================
-- INDEX complementaires (performance des requetes analytiques)
-- ==========================================================

CREATE INDEX idx_fact_sales_date ON fact_sales(date_id);
CREATE INDEX idx_fact_sales_product ON fact_sales(product_id);
CREATE INDEX idx_fact_inventory_date ON fact_inventory(date_id);
CREATE INDEX idx_fact_inventory_product ON fact_inventory(product_id);
CREATE INDEX idx_fact_production_date ON fact_production(date_id);
CREATE INDEX idx_fact_production_machine ON fact_production(machine_id);
CREATE INDEX idx_fact_energy_timestamp ON fact_energy(timestamp);
CREATE INDEX idx_fact_energy_machine ON fact_energy(machine_id);
CREATE INDEX idx_fact_machine_sensor_timestamp ON fact_machine_sensor(timestamp);
CREATE INDEX idx_fact_machine_sensor_machine ON fact_machine_sensor(machine_id);
CREATE INDEX idx_fact_transport_date ON fact_transport(date_id);
CREATE INDEX idx_fact_anomaly_timestamp ON fact_anomaly(timestamp);
CREATE INDEX idx_fact_anomaly_domain ON fact_anomaly(domain);

-- ==========================================================
-- Fin du script
-- ==========================================================

-- Alter existing trips table to add missing columns for ride-booking system

ALTER TABLE trips
ADD COLUMN driver_id INT NULL AFTER user_id,
ADD COLUMN pickup_address VARCHAR(500) NULL AFTER driver_id,
ADD COLUMN pickup_lat DECIMAL(10,8) NULL AFTER pickup_address,
ADD COLUMN pickup_lng DECIMAL(11,8) NULL AFTER pickup_lat,
ADD COLUMN dropoff_address VARCHAR(500) NULL AFTER pickup_lng,
ADD COLUMN dropoff_lat DECIMAL(10,8) NULL AFTER dropoff_address,
ADD COLUMN dropoff_lng DECIMAL(11,8) NULL AFTER dropoff_lat,
ADD COLUMN requested_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP AFTER dropoff_lng,
ADD COLUMN assigned_at TIMESTAMP NULL AFTER requested_at,
ADD COLUMN completed_at TIMESTAMP NULL AFTER assigned_at,
ADD COLUMN eta_minutes INT NULL AFTER duration_minutes,
ADD CONSTRAINT fk_trips_driver_id FOREIGN KEY (driver_id) REFERENCES users(id);

-- Update existing data to populate new columns from old ones
UPDATE trips SET
    pickup_address = start_location,
    dropoff_address = end_location,
    requested_at = start_time,
    completed_at = end_time
WHERE pickup_address IS NULL;

-- Add indexes
CREATE INDEX idx_trips_driver_id ON trips(driver_id);
CREATE INDEX idx_trips_requested_at ON trips(requested_at);
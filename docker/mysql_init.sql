CREATE DATABASE IF NOT EXISTS health_platform;
USE health_platform;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    role ENUM('admin', 'doctor', 'patient') DEFAULT 'patient',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_email (email),
    INDEX idx_role (role)
);

CREATE TABLE IF NOT EXISTS patients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    patient_code VARCHAR(20) UNIQUE NOT NULL,
    age INT NOT NULL,
    gender ENUM('male', 'female', 'other') NOT NULL,
    blood_group VARCHAR(5),
    contact_number VARCHAR(15),
    address TEXT,
    medical_history JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    INDEX idx_patient_code (patient_code)
);

CREATE TABLE IF NOT EXISTS health_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT NOT NULL,
    recorded_by INT,
    blood_pressure_systolic INT,
    blood_pressure_diastolic INT,
    heart_rate INT,
    blood_glucose FLOAT,
    bmi FLOAT,
    cholesterol_total FLOAT,
    cholesterol_hdl FLOAT,
    cholesterol_ldl FLOAT,
    hemoglobin FLOAT,
    temperature FLOAT,
    oxygen_saturation FLOAT,
    notes TEXT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
    FOREIGN KEY (recorded_by) REFERENCES users(id) ON DELETE SET NULL,
    INDEX idx_patient_recorded (patient_id, recorded_at)
);

CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT NOT NULL,
    health_record_id INT,
    risk_score FLOAT NOT NULL,
    risk_level ENUM('low', 'medium', 'high', 'critical') NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    shap_values JSON,
    top_risk_factors JSON,
    prediction_metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
    FOREIGN KEY (health_record_id) REFERENCES health_records(id) ON DELETE SET NULL,
    INDEX idx_patient_predictions (patient_id, created_at)
);

CREATE TABLE IF NOT EXISTS triage_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    symptoms TEXT NOT NULL,
    rag_response TEXT,
    urgency_level ENUM('routine', 'soon', 'urgent', 'emergency') NOT NULL,
    retrieved_docs JSON,
    session_metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS medical_knowledge (
    id INT AUTO_INCREMENT PRIMARY KEY,
    category VARCHAR(100) NOT NULL,
    condition_name VARCHAR(255) NOT NULL,
    symptoms TEXT NOT NULL,
    risk_factors TEXT,
    description TEXT,
    urgency_guidelines TEXT,
    embedding_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_category (category),
    FULLTEXT INDEX ft_search (condition_name, symptoms, risk_factors)
);

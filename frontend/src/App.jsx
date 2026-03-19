import React, { useState } from 'react';
import { Activity, Beaker, Heart, Brain, Ruler, Droplet, User, Scale } from 'lucide-react';
import { predictRisk } from './api';

const App = () => {
  const [formData, setFormData] = useState({
    pregnancies: '',
    glucose: '',
    bloodpressure: '',
    skinthickness: '',
    insulin: '',
    bmi: '',
    diabetespedigreefunction: '',
    age: ''
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const inputs = [
    { name: 'age', label: 'Age (years)', min: 1, icon: <User size={18} /> },
    { name: 'pregnancies', label: 'Pregnancies', min: 0, icon: <Heart size={18} /> },
    { name: 'glucose', label: 'Glucose Level (mg/dL)', min: 1, icon: <Droplet size={18} /> },
    { name: 'bloodpressure', label: 'Blood Pressure (mm Hg)', min: 1, icon: <Activity size={18} /> },
    { name: 'skinthickness', label: 'Skin Thickness (mm)', min: 1, icon: <Ruler size={18} /> },
    { name: 'insulin', label: 'Insulin (mu U/ml)', min: 0, icon: <Beaker size={18} /> },
    { name: 'bmi', label: 'BMI (kg/m²)', min: 1, step: "0.1", icon: <Scale size={18} /> },
    { name: 'diabetespedigreefunction', label: 'Diabetes Pedigree', min: 0, step: "0.01", icon: <Brain size={18} /> },
  ];

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value === '' ? '' : Number(value)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    // Validate if any field is empty
    for (const key in formData) {
      if (formData[key] === '') {
        setError("Please fill in all medical fields correctly.");
        setLoading(false);
        return;
      }
    }

    try {
      const response = await predictRisk(formData);
      setResult(response);
    } catch (err) {
      setError("Failed to connect to the predictive engine. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setFormData({
      pregnancies: '',
      glucose: '',
      bloodpressure: '',
      skinthickness: '',
      insulin: '',
      bmi: '',
      diabetespedigreefunction: '',
      age: ''
    });
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>MediGuide AI</h1>
        <p>Advanced Diabetes Risk Assessment System</p>
      </header>

      <main className="glass-panel">
        {!result ? (
          <form onSubmit={handleSubmit} className="form-grid">
            {inputs.map((input) => (
              <div key={input.name} className="input-group">
                <label style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  {input.icon} {input.label}
                </label>
                <input
                  type="number"
                  name={input.name}
                  value={formData[input.name]}
                  onChange={handleChange}
                  min={input.min}
                  step={input.step || "1"}
                  placeholder={`Enter ${input.label.split(' ')[0].toLowerCase()}`}
                  required
                />
              </div>
            ))}

            {error && <div className="error-message">{error}</div>}

            <button type="submit" className="submit-btn" disabled={loading}>
              {loading ? <div className="loader"></div> : (
                <>
                  <Activity size={20} />
                  Analyze Patient Data
                </>
              )}
            </button>
          </form>
        ) : (
          <div className="result-card">
            <h2>Clinical Risk Assessment</h2>
            
            <div className={`risk-level ${result.risk_level}`}>
              {result.risk_level} Risk
            </div>
            
            <div className="probability-container">
              <div className="prob-label">
                <span>Confidence Probability</span>
                <span>{(result.risk_probability * 100).toFixed(1)}%</span>
              </div>
              <div className="probability-bar">
                <div 
                  className={`probability-fill fill-${result.risk_level}`} 
                  style={{ width: `${result.risk_probability * 100}%` }}
                ></div>
              </div>
            </div>

            <div className="reset-wrapper">
              <button className="reset-btn" onClick={handleReset}>
                New Assessment
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;

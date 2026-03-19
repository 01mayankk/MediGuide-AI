import React, { useState, useEffect, useRef } from 'react';
import { Activity, Beaker, Heart, Brain, Ruler, Droplet, User, Scale, TrendingUp, Calendar, History, Sun, Moon, ChevronDown } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { predictRisk } from './api';

// --- Custom Premium Glassmorphism Dropdown ---
const CustomDropdown = ({ value, onChange, name, options, placeholder }) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const selectedOption = options.find(opt => opt.value === value) || options.find(opt => opt.value === String(value));

  return (
    <div className="custom-dropdown-container" ref={dropdownRef}>
      <div 
        className={`custom-dropdown-trigger neumorphic-input ${isOpen ? 'active glow-border' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
        role="button"
        tabIndex={0}
      >
        <span className="dropdown-text">
          {selectedOption ? selectedOption.label : placeholder}
        </span>
        <ChevronDown size={18} className={`dropdown-arrow ${isOpen ? 'open' : ''}`} />
      </div>
      {isOpen && (
        <ul className="custom-dropdown-menu glass-panel slide-down-anim">
          {options.map((opt, i) => (
            <li 
              key={i} 
              className={`dropdown-item ${String(opt.value) === String(value) ? 'selected' : ''}`}
              onClick={() => {
                onChange({ target: { name, value: opt.value } });
                setIsOpen(false);
              }}
            >
              {opt.label}
              {String(opt.value) === String(value) && <div className="selected-indicator"></div>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};


const App = () => {
  const [theme, setTheme] = useState(localStorage.getItem('mediguide_theme') || 'dark');
  
  // Custom Profiles System
  const [profiles, setProfiles] = useState(() => {
    const saved = localStorage.getItem('mediguide_profiles_v2');
    if (saved) return JSON.parse(saved);
    return [{ id: 'guest', name: 'Guest Profile', dob: '1990-01-01', gender: 'Female' }];
  });
  
  const [currentProfileId, setCurrentProfileId] = useState(() => {
    return localStorage.getItem('mediguide_user_id') || profiles[0].id;
  });

  const currentProfile = profiles.find(p => p.id === currentProfileId) || profiles[0];

  const [formData, setFormData] = useState({
    pregnancies: '',
    glucose: '',
    bloodpressure: '',
    skinthickness: '',
    insulin: '',
    bmi: '',
    diabetespedigreefunction: ''
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showProfileModal, setShowProfileModal] = useState(false);
  const [newProfile, setNewProfile] = useState({ name: '', dob: '', gender: 'Female' });
  
  // History State
  const [history, setHistory] = useState(() => {
    const saved = localStorage.getItem(`mediguide_history_${currentProfileId}`);
    return saved ? JSON.parse(saved) : [];
  });

  // Calculate Age dynamically
  const calculateAge = (dob) => {
    if (!dob) return 0;
    const diff = Date.now() - new Date(dob).getTime();
    const ageDate = new Date(diff); 
    return Math.abs(ageDate.getUTCFullYear() - 1970);
  };

  useEffect(() => {
    localStorage.setItem('mediguide_user_id', currentProfileId);
    const saved = localStorage.getItem(`mediguide_history_${currentProfileId}`);
    setHistory(saved ? JSON.parse(saved) : []);
    setResult(null); // Clear result on user switch
  }, [currentProfileId]);

  useEffect(() => {
    localStorage.setItem(`mediguide_history_${currentProfileId}`, JSON.stringify(history));
  }, [history, currentProfileId]);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('mediguide_theme', theme);
  }, [theme]);

  const toggleTheme = () => setTheme(prev => prev === 'light' ? 'dark' : 'light');

  const handleCreateProfile = (e) => {
    e.preventDefault();
    const newId = Date.now().toString();
    const profile = { id: newId, ...newProfile };
    const updated = [...profiles, profile];
    setProfiles(updated);
    localStorage.setItem('mediguide_profiles_v2', JSON.stringify(updated));
    setCurrentProfileId(newId);
    setShowProfileModal(false);
    setNewProfile({ name: '', dob: '', gender: 'Female' });
  };

  const numberInputs = [
    { name: 'glucose', label: 'Blood Sugar Level', helper: 'Usually checked after fasting (mg/dL)', min: 1, icon: <Droplet size={18} /> },
    { name: 'bloodpressure', label: 'Blood Pressure', helper: 'The bottom number of your reading (Diastolic in mm Hg)', min: 1, icon: <Activity size={18} /> },
    { name: 'skinthickness', label: 'Skin Thickness', helper: 'Body fat estimate on the back of your arm (in mm)', min: 1, icon: <Ruler size={18} /> },
    { name: 'insulin', label: 'Insulin Level', helper: 'Shows how your body handles sugar (mu U/ml)', min: 0, icon: <Beaker size={18} /> },
    { name: 'bmi', label: 'Body Mass Index (BMI)', helper: 'Calculated from your weight and height (kg/m²)', min: 1, step: "0.1", icon: <Scale size={18} /> },
  ];

  const handleChange = (e) => {
    const { name, value } = e.target;
    if (value === '__ADD_NEW__') {
      setShowProfileModal(true);
    } else {
      setFormData(prev => ({ ...prev, [name]: value }));
    }
  };

  const handleProfileChange = (e) => {
    if (e.target.value === '__ADD_NEW__') setShowProfileModal(true);
    else setCurrentProfileId(e.target.value);
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    const requiredFields = ['glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi', 'diabetespedigreefunction'];
    if (currentProfile.gender === 'Female') requiredFields.push('pregnancies');

    for (const key of requiredFields) {
      if (formData[key] === '') {
        setError("Please fill in all medical fields correctly.");
        setLoading(false);
        return;
      }
    }

    try {
      const age = calculateAge(currentProfile.dob);
      const payload = {
        pregnancies: currentProfile.gender === 'Male' ? 0 : Number(formData.pregnancies),
        glucose: Number(formData.glucose),
        bloodpressure: Number(formData.bloodpressure),
        skinthickness: Number(formData.skinthickness),
        insulin: Number(formData.insulin),
        bmi: Number(formData.bmi),
        diabetespedigreefunction: Number(formData.diabetespedigreefunction),
        age: age,
      };

      const response = await predictRisk(payload);
      setResult(response);
      
      const newRecord = {
        date: new Date().toLocaleDateString() + ' ' + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        probability: Number((response.risk_probability * 100).toFixed(1)),
        risk_level: response.risk_level,
        metrics: { ...payload }
      };
      setHistory(prev => [...prev, newRecord]);
    } catch (err) {
      setError("Failed to connect to the predictive engine. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const chartData = history.map((item, index) => ({
    name: `Test ${index + 1}`,
    rawDate: item.date,
    Risk: item.probability
  }));

  // Options for dropdowns
  const profileOptions = [
    ...profiles.map(p => ({ value: p.id, label: p.name })),
    { value: '__ADD_NEW__', label: '+ Create Profile...' }
  ];

  const pregnancyOptions = [
    { value: "", label: "Select amount" },
    ...[...Array(15).keys()].map(num => ({ value: num, label: `${num}${num === 14 ? '+' : ''}` }))
  ];

  const familyHistoryOptions = [
    { value: "", label: "Select option" },
    { value: "0.1", label: "No family history" },
    { value: "0.4", label: "Some family history (e.g. uncle, aunt)" },
    { value: "0.8", label: "Strong family history (e.g. parent, sibling)" },
    { value: "1.2", label: "Very strong family history (multiple close relatives)" }
  ];

  return (
    <div className="medical-app">
      {/* Dynamic Background Particles via CSS */}
      <div className="bg-particles"></div>

      {/* Navbar */}
      <nav className="top-navbar glass-panel">
        <div className="nav-brand">
          <div className="brand-icon-wrapper">
            <Activity size={24} className="brand-icon" />
          </div>
          <h1>MediGuide AI</h1>
        </div>
        <div className="nav-controls">
          <div className="profile-selector neumorphic" style={{zIndex: 60}}>
            <User size={18} className="text-primary" style={{marginRight: '0.5rem'}} />
            <div style={{width: '200px'}}>
              <CustomDropdown
                name="profileSelect"
                value={currentProfileId}
                onChange={handleProfileChange}
                options={profileOptions}
                placeholder="Select Profile"
              />
            </div>
          </div>
          <button className="theme-toggle neumorphic-btn" onClick={toggleTheme} aria-label="Toggle Theme">
            {theme === 'light' ? <Moon size={18} /> : <Sun size={18} />}
          </button>
        </div>
      </nav>

      {/* Hero Banner */}
      <section className="hero-banner fade-in">
        <div className="hero-content-wrapper">
          <div className="hero-text">
            <h2 className="gradient-text">Predict. Prevent. Protect.</h2>
            <p className="hero-subtext">Welcome back, <strong>{currentProfile.name}</strong>. Let's check your vitals and secure your health today.</p>
            <div className="hero-tags">
              <span className="hero-tag glass-pill"><Calendar size={14} /> Age: {calculateAge(currentProfile.dob)}</span>
              <span className="hero-tag glass-pill"><User size={14} /> Gender: {currentProfile.gender}</span>
            </div>
          </div>
          <div className="hero-image-container">
            <div className="image-glow-backdrop"></div>
            <img src="/hero-v2.png" alt="Clinical Assistant" className="hero-illustration float-anim" />
          </div>
        </div>
      </section>

      {/* Main Content Layout */}
      <main className="content-grid">
        
        {/* Left Column: Form with Sitting Doctor */}
        <div className="input-section-wrapper slide-up">
          <div className="sitting-doctor-container">
            <div className="dr-tooltip glow-box fade-in">I'm here to help you!</div>
            <img src="/sitting-v2.png" alt="Friendly Assistant" className="sitting-doctor" />
          </div>

          <div className="clinical-card form-card glass-panel neumorphic-inset">
            <div className="card-header">
              <h2><Activity className="header-icon" /> Health Diagnostics</h2>
              <p className="card-subtitle">Input your recent physiological metrics smoothly</p>
            </div>
            
            <form onSubmit={handleSubmit} className="clinical-form">
              {currentProfile.gender === 'Female' && (
                <div className="input-group">
                  <label><Heart size={16} className="input-icon" /> Pregnancies</label>
                  <span className="helper-text">Number of times you have been pregnant</span>
                  <div className="input-wrapper" style={{zIndex: 50}}>
                    <CustomDropdown
                      name="pregnancies"
                      value={formData.pregnancies}
                      onChange={handleChange}
                      options={pregnancyOptions}
                      placeholder="Select amount"
                    />
                  </div>
                </div>
              )}

              {numberInputs.map((input) => (
                <div key={input.name} className="input-group">
                  <label>
                    <span className="input-icon-wrapper">{input.icon}</span> {input.label}
                  </label>
                  {input.helper && <span className="helper-text">{input.helper}</span>}
                  <div className="input-wrapper">
                    <input
                      className="clinical-input neumorphic-input"
                      type="number"
                      name={input.name}
                      value={formData[input.name]}
                      onChange={handleChange}
                      min={input.min}
                      step={input.step || "1"}
                      placeholder={`e.g. 120`}
                      required
                    />
                    <div className="input-focus-glow"></div>
                  </div>
                </div>
              ))}

              <div className="input-group">
                <label><Brain size={16} className="input-icon" /> Family History</label>
                <span className="helper-text">Does anyone in your immediate family have diabetes?</span>
                <div className="input-wrapper" style={{zIndex: 40}}>
                  <CustomDropdown
                    name="diabetespedigreefunction"
                    value={formData.diabetespedigreefunction}
                    onChange={handleChange}
                    options={familyHistoryOptions}
                    placeholder="Select option"
                  />
                </div>
              </div>

              {error && <div className="error-message glow-box-danger">{error}</div>}

              <button type="submit" className="primary-btn submit-btn glowing-btn" disabled={loading}>
                {loading ? <div className="loader"></div> : 'Analyze Diagnostics'}
              </button>
            </form>
          </div>
        </div>

        {/* Right Column: Results & Trends */}
        <div className="metrics-column">
          {result && (
            <div className="clinical-card result-card glass-panel fade-in slide-up">
              <div className="card-header highlight">
                <h2>Diagnostic Result</h2>
              </div>
              <div className="result-content">
                <div className={`risk-badge glow-risk-${result.risk_level}`}>
                  {result.risk_level} Risk Level
                </div>
                
                <div className="prob-meter neumorphic-inset-box">
                  <div className="prob-meter-labels">
                    <span>AI Confidence Score</span>
                    <strong className="gradient-text">{(result.risk_probability * 100).toFixed(1)}%</strong>
                  </div>
                  <div className="prob-meter-track neumorphic-track">
                    <div 
                      className={`prob-meter-fill fill-glow-${result.risk_level}`} 
                      style={{ width: `${result.risk_probability * 100}%` }}
                    >
                      <div className="fill-highlight"></div>
                    </div>
                  </div>
                </div>

                <div className={`clinical-feedback alert-${result.risk_level} glass-alert`}>
                  {result.risk_level === 'Low' && (
                    <p><strong className="alert-title">✅ Great news.</strong> Your indicators point to a low clinical risk. Keep maintaining your beautifully healthy routine.</p>
                  )}
                  {result.risk_level === 'Medium' && (
                    <p><strong className="alert-title">⚠️ Moderate Alert.</strong> Certain metrics are elevated. Consider scheduling a routine evaluation focusing on dietary improvements.</p>
                  )}
                  {result.risk_level === 'High' && (
                    <p><strong className="alert-title text-danger">🚨 Clinical Attention Advised.</strong> Your metrics indicate high risk. We strongly recommend consulting a healthcare provider immediately to discuss preventative measures.</p>
                  )}
                </div>

                <div className="action-row">
                  <button type="button" className="secondary-btn neumorphic-btn" onClick={() => setResult(null)}>Clear Result</button>
                  {(result.risk_level === 'Medium' || result.risk_level === 'High') && (
                    <a href="https://www.google.com/maps/search/hospitals+near+me" target="_blank" rel="noreferrer" className="primary-btn danger-btn glowing-btn" style={{textDecoration: 'none'}}>
                      Find Nearby Hospitals
                    </a>
                  )}
                </div>
              </div>
            </div>
          )}

          <div className="clinical-card trends-card glass-panel fade-in">
            <div className="card-header">
              <h2><TrendingUp className="header-icon" /> Historical Trends</h2>
            </div>
            
            {history.length > 0 ? (
              <div className="chart-wrapper">
                <ResponsiveContainer width="100%" height={260}>
                  <AreaChart data={chartData} margin={{ top: 15, right: 10, left: -20, bottom: 0 }}>
                    <defs>
                      <linearGradient id="colorRisk" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="var(--accent-cyan)" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="var(--accent-blue)" stopOpacity={0}/>
                      </linearGradient>
                      <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
                        <feGaussianBlur stdDeviation="4" result="blur" />
                        <feComposite in="SourceGraphic" in2="blur" operator="over" />
                      </filter>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" className="chart-grid" stroke="var(--border-light)" vertical={false} />
                    <XAxis dataKey="name" stroke="var(--text-muted)" fontSize={12} tickLine={false} axisLine={false} dy={10} />
                    <YAxis stroke="var(--text-muted)" fontSize={12} tickLine={false} axisLine={false} dx={-10} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: 'var(--card-bg)', backdropFilter: 'blur(10px)', borderColor: 'var(--border-light)', borderRadius: '12px', color: 'var(--text-main)', boxShadow: 'var(--shadow-lg)' }}
                      itemStyle={{ color: 'var(--accent-cyan)', fontWeight: 'bold' }}
                      cursor={{ stroke: 'var(--accent-cyan)', strokeWidth: 1, strokeDasharray: '4 4' }}
                    />
                    <Area type="monotone" dataKey="Risk" stroke="var(--accent-cyan)" strokeWidth={4} fillOpacity={1} fill="url(#colorRisk)" filter="url(#glow)" activeDot={{ r: 6, fill: 'var(--bg-body)', stroke: 'var(--accent-cyan)', strokeWidth: 3 }} />
                  </AreaChart>
                </ResponsiveContainer>
                
                {history.length > 1 && (
                  <div className="trend-insights neumorphic-inset">
                    <h4 className="gradient-text">Detailed Attribute Analysis</h4>
                    <ul className="detailed-trends">
                      {(() => {
                        const curr = history[history.length - 1].metrics;
                        const prev = history[history.length - 2].metrics;
                        const diffs = [];
                        if (curr && prev) {
                          // Glucose
                          if (curr.glucose > prev.glucose) diffs.push({ text: `Blood Sugar increased by ${(curr.glucose - prev.glucose).toFixed(1)} mg/dL.`, isGood: false, advice: 'Consider monitoring your carbohydrate intake.' });
                          else if (curr.glucose < prev.glucose) diffs.push({ text: `Blood Sugar decreased by ${(prev.glucose - curr.glucose).toFixed(1)} mg/dL.`, isGood: true, advice: 'Excellent metabolic control!' });
                          
                          // Blood Pressure
                          if (curr.bloodpressure > prev.bloodpressure) diffs.push({ text: `Blood Pressure increased by ${(curr.bloodpressure - prev.bloodpressure).toFixed(1)} mmHg.`, isGood: false, advice: 'Monitor sodium intake and stress levels.' });
                          else if (curr.bloodpressure < prev.bloodpressure) diffs.push({ text: `Blood Pressure decreased by ${(prev.bloodpressure - curr.bloodpressure).toFixed(1)} mmHg.`, isGood: true, advice: 'Great job maintaining cardiovascular health!' });

                          // Skin Thickness
                          if (curr.skinthickness > prev.skinthickness) diffs.push({ text: `Skin Thickness measure increased by ${(curr.skinthickness - prev.skinthickness).toFixed(1)} mm.`, isGood: false, advice: 'May indicate an increase in body fat.' });
                          else if (curr.skinthickness < prev.skinthickness) diffs.push({ text: `Skin Thickness measure decreased by ${(prev.skinthickness - curr.skinthickness).toFixed(1)} mm.`, isGood: true, advice: 'Positive reduction in peripheral fat.' });

                          // Insulin
                          if (curr.insulin > prev.insulin) diffs.push({ text: `Insulin level increased by ${(curr.insulin - prev.insulin).toFixed(1)} mu U/ml.`, isGood: false, advice: 'Could be a sign of increased insulin resistance.' });
                          else if (curr.insulin < prev.insulin && prev.insulin !== 0 && curr.insulin !== 0) diffs.push({ text: `Insulin level decreased by ${(prev.insulin - curr.insulin).toFixed(1)} mu U/ml.`, isGood: true, advice: 'Your body is handling sugar more efficiently!' });

                          // BMI
                          if (curr.bmi > prev.bmi) diffs.push({ text: `BMI increased by ${(curr.bmi - prev.bmi).toFixed(1)}.`, isGood: false, advice: 'Regular cardiovascular activity is recommended.' });
                          else if (curr.bmi < prev.bmi) diffs.push({ text: `BMI decreased by ${(prev.bmi - curr.bmi).toFixed(1)}.`, isGood: true, advice: 'Fantastic progress on weight management!' });

                          // Pregnancies
                          if (curr.pregnancies > prev.pregnancies) diffs.push({ text: `Pregnancies recorded increased from ${prev.pregnancies} to ${curr.pregnancies}.`, isGood: null, advice: 'Ensure regular prenatal checkups.' });
                          
                          // Pedigree
                          if (curr.diabetespedigreefunction !== prev.diabetespedigreefunction) diffs.push({ text: `Family History Score changed to ${curr.diabetespedigreefunction}.`, isGood: null, advice: 'Updated genetic baseline.' });
                        }
                        
                        if (diffs.length === 0) return <li><div className="trend-item neutral glass-alert"><strong>No changes detected.</strong><p>Your vital signs are exactly the same as your last test.</p></div></li>;
                        
                        return diffs.map((d, i) => (
                          <li key={i}>
                            <div className={`trend-item ${d.isGood === true ? 'good' : d.isGood === false ? 'bad' : 'neutral'} glass-alert`}>
                              <div className="trend-header">
                                <span className="trend-status">{d.isGood === true ? '✅ Improved:' : d.isGood === false ? '⚠️ Attention:' : 'ℹ️ Note:'}</span>
                                <strong>{d.text}</strong>
                              </div>
                              <p className="trend-advice">{d.advice}</p>
                            </div>
                          </li>
                        ));
                      })()}
                    </ul>
                  </div>
                )}
              </div>
            ) : (
              <div className="empty-history neumorphic-inset">
                <History size={48} className="empty-icon" />
                <p>No diagnostics recorded for <strong>{currentProfile.name}</strong>. Begin an assessment to establish a health baseline.</p>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Profile Modal */}
      {showProfileModal && (
        <div className="modal-overlay glass-panel">
          <div className="modal-card neumorphic">
            <h3 className="gradient-text">Add New Profile</h3>
            <form onSubmit={handleCreateProfile}>
              <div className="input-group">
                <label>Full Name</label>
                <div className="input-wrapper">
                  <input className="clinical-input neumorphic-input" type="text" required value={newProfile.name} onChange={e => setNewProfile({...newProfile, name: e.target.value})} placeholder="e.g. Grandma, John" />
                </div>
              </div>
              <div className="input-group">
                <label>Date of Birth</label>
                <div className="input-wrapper">
                  <input className="clinical-input neumorphic-input" type="date" required value={newProfile.dob} onChange={e => setNewProfile({...newProfile, dob: e.target.value})} />
                </div>
                <span className="helper-text">Used to automatically calculate precise age</span>
              </div>
              <div className="input-group">
                <label>Biological Gender</label>
                <div className="input-wrapper" style={{zIndex: 60}}>
                  <CustomDropdown
                     name="gender"
                     value={newProfile.gender}
                     onChange={(e) => setNewProfile({...newProfile, gender: e.target.value})}
                     options={[{value: 'Female', label: 'Female'}, {value: 'Male', label: 'Male'}]}
                     placeholder="Select Gender"
                  />
                </div>
              </div>
              <div className="modal-actions">
                <button type="button" className="secondary-btn neumorphic-btn" onClick={() => setShowProfileModal(false)}>Cancel</button>
                <button type="submit" className="primary-btn glowing-btn">Create Profile</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;

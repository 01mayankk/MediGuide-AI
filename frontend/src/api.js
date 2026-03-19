import axios from 'axios';

const API_BASE_URL = 'https://01mayankk-mediguide-ai.hf.space';

export const predictRisk = async (patientData) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/predict`, patientData);
    return response.data;
  } catch (error) {
    console.error("API Error: Failed to predict risk", error);
    throw error;
  }
};

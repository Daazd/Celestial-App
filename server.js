const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');

const app = express();
const port = process.env.PORT || 5000;

app.use(cors());
app.use(bodyParser.json());

// Connect to MongoDB
mongoose.connect('mongodb://localhost/celestial_db', { useNewUrlParser: true, useUnifiedTopology: true });

// Define schemas
const celestialBodySchema = new mongoose.Schema({
  name: String,
  type: String,
  coordinates: {
    rightAscension: Number,
    declination: Number
  },
  description: String
});

const CelestialBody = mongoose.model('CelestialBody', celestialBodySchema);

// API endpoints
app.get('/api/celestial-bodies', async (req, res) => {
  const bodies = await CelestialBody.find();
  res.json(bodies);
});

app.get('/api/celestial-bodies/:id', async (req, res) => {
  const body = await CelestialBody.findById(req.params.id);
  res.json(body);
});

app.post('/api/celestial-bodies', async (req, res) => {
  const newBody = new CelestialBody(req.body);
  await newBody.save();
  res.json(newBody);
});

// Chatbot endpoint
app.post('/chat', (req, res) => {
    const userInput = req.body.message;
    
    // Use 'python3' instead of 'python'
    const python = spawn('python3', ['chatbot.py', userInput]);
    
    let chatbotResponse = '';
    let errorOccurred = false;
  
    // Collect data from script
    python.stdout.on('data', (data) => {
      chatbotResponse += data.toString();
    });
  
    // Handle potential errors
    python.stderr.on('data', (data) => {
      console.error(`Error from Python script: ${data}`);
      errorOccurred = true;
    });
  
    // Send response when script finishes
    python.on('close', (code) => {
      console.log(`Child process exited with code ${code}`);
      if (errorOccurred) {
        res.status(500).json({ error: 'An error occurred while processing your request.' });
      } else {
        res.json({ response: chatbotResponse.trim() });
      }
    });
  });

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
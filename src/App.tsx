import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import Home from './components/Home';
import StarCoordinates from './components/StarCoordinates';
import Chatbot from './components/Chatbot';
import StargazingLocations from './components/StargazingLocations';
import './App.css';  // Import the CSS file here

const App: React.FC = () => {
  return (
    <Router>
      <div className="App">
      <nav className="menu">
          <ul className="flex justify-center space-x-4">
            <li><Link to="/" className="text-white hover:text-blue-300">Home</Link></li>
            <li><Link to="/coordinates" className="text-white hover:text-blue-300">Star Coordinates</Link></li>
            <li><Link to="/chatbot" className="text-white hover:text-blue-300">Celestial Chatbot</Link></li>
            <li><Link to="/locations" className="text-white hover:text-blue-300">Stargazing Locations</Link></li>
          </ul>
        </nav>

        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/coordinates" element={<StarCoordinates />} />
          <Route path="/chatbot" element={<Chatbot />} />
          <Route path="/locations" element={<StargazingLocations />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
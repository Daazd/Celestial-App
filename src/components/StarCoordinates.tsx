import React, { useState } from 'react';
import axios from 'axios';

interface Coordinates {
  rightAscension: number;
  declination: number;
}

interface CelestialBody {
  name: string;
  coordinates: Coordinates;
}

const StarCoordinates: React.FC = () => {
  const [starName, setStarName] = useState<string>('');
  const [coordinates, setCoordinates] = useState<Coordinates | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      const response = await axios.get<CelestialBody[]>(`http://localhost:5000/api/celestial-bodies`, {
        params: { name: starName }
      });
      if (response.data.length > 0) {
        setCoordinates(response.data[0].coordinates);
      } else {
        setError('No star found with that name. Please try another.');
      }
    } catch (error) {
      setError('Unable to fetch star coordinates. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Star Coordinates Lookup</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={starName}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setStarName(e.target.value)}
          placeholder="Enter star name"
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Loading...' : 'Lookup'}
        </button>
      </form>
      {error && <p style={{ color: 'red' }}>{error}</p>}
      {coordinates && (
        <div>
          <h3>{starName}</h3>
          <p>Right Ascension: {coordinates.rightAscension}</p>
          <p>Declination: {coordinates.declination}</p>
        </div>
      )}
    </div>
  );
};

export default StarCoordinates;
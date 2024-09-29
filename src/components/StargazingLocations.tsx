import React, { useState, useEffect } from 'react';
import axios from 'axios';

interface Location {
  latitude: number;
  longitude: number;
}

interface StargazingSpot {
  name: string;
  distance: number;
}

const StargazingLocations: React.FC = () => {
  const [location, setLocation] = useState<Location | null>(null);
  const [locations, setLocations] = useState<StargazingSpot[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position: GeolocationPosition) => {
          setLocation({
            latitude: position.coords.latitude,
            longitude: position.coords.longitude
          });
        },
        (error: GeolocationPositionError) => {
          setError("Unable to retrieve your location. Please enable location services.");
          setLoading(false);
        }
      );
    } else {
      setError("Geolocation is not supported by your browser.");
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (location) {
      fetchStargazingLocations();
    }
  }, [location]);

  const fetchStargazingLocations = async () => {
    if (!location) return;

    try {
      const response = await axios.get<StargazingSpot[]>(`http://localhost:5000/api/stargazing-locations`, {
        params: {
          lat: location.latitude,
          lon: location.longitude
        }
      });
      setLocations(response.data);
      setLoading(false);
    } catch (error) {
      setError('Failed to fetch stargazing locations. Please try again later.');
      setLoading(false);
    }
  };

  if (loading) return <p>Loading...</p>;
  if (error) return <p>{error}</p>;

  return (
    <div>
      <h2>Best Stargazing Locations Near You</h2>
      {location && (
        <p>Your location: {location.latitude.toFixed(4)}, {location.longitude.toFixed(4)}</p>
      )}
      <ul>
        {locations.map((loc, index) => (
          <li key={index}>{loc.name} - {loc.distance} miles</li>
        ))}
      </ul>
    </div>
  );
};

export default StargazingLocations;
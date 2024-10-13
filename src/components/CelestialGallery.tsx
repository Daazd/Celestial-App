import React, { useState, useEffect, useMemo } from 'react';
import styles from './CelestialGallery.module.css';

interface CelestialBody {
  _id: string;
  name: string;
  description: string;
  keywords: string;
  image_url: string;
  date: string;
}

const CelestialGallery: React.FC = () => {
  const [celestialBodies, setCelestialBodies] = useState<CelestialBody[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedBody, setSelectedBody] = useState<CelestialBody | null>(null);
  const [showBackToTop, setShowBackToTop] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    fetchCelestialBodies();

    const handleScroll = () => {
      setShowBackToTop(window.scrollY > 300);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const fetchCelestialBodies = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/celestial-bodies');
      if (!response.ok) {
        throw new Error('Failed to fetch celestial bodies');
      }
      const data = await response.json();
      const filteredData = data.filter((body: CelestialBody) => body.image_url);
      setCelestialBodies(filteredData);
      setLoading(false);
    } catch (err) {
      setError('Error fetching celestial bodies. Please try again later.');
      setLoading(false);
    }
  };

  const filteredBodies = useMemo(() => {
    return celestialBodies.filter((body) => {
      const searchContent = `${body.name} ${body.description} ${body.keywords}`.toLowerCase();
      return searchContent.includes(searchTerm.toLowerCase());
    });
  }, [celestialBodies, searchTerm]);

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleBodyClick = (body: CelestialBody) => {
    setSelectedBody(body);
  };

  const handleCloseModal = () => {
    setSelectedBody(null);
  };

  const handleImageError = (event: React.SyntheticEvent<HTMLImageElement, Event>) => {
    event.currentTarget.src = '/placeholder-image.jpg';
  };

  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(event.target.value);
  };

  if (loading) {
    return <div className={styles.loadingText}>Loading celestial bodies...</div>;
  }

  if (error) {
    return <div className={styles.errorText}>{error}</div>;
  }

  return (
    <div className={styles.galleryContainer}>
      <h1 className={styles.galleryTitle}>Celestial Gallery</h1>
      <div className={styles.searchContainer}>
        <input
          type="text"
          placeholder="Search celestial bodies..."
          value={searchTerm}
          onChange={handleSearchChange}
          className={styles.searchInput}
        />
      </div>
      <div className={styles.galleryGrid}>
        {filteredBodies.map((body) => (
          <div
            key={body._id}
            className={styles.galleryItem}
            onClick={() => handleBodyClick(body)}
          >
            <div className={styles.galleryItemImageContainer}>
              <img
                src={body.image_url || '/placeholder-image.jpg'}
                alt={body.name}
                className={styles.galleryItemImage}
                onError={handleImageError}
              />
            </div>
            <div className={styles.galleryItemInfo}>
              <h2 className={styles.galleryItemTitle}>{body.name}</h2>
            </div>
          </div>
        ))}
      </div>

      {showBackToTop && (
        <button 
          className={styles.backToTopButton}
          onClick={scrollToTop}
        >
          ↑
        </button>
      )}

      {selectedBody && (
        <div className={styles.modal} onClick={handleCloseModal}>
          <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
            <button
              onClick={handleCloseModal}
              className={styles.modalCloseButton}
            >
              ×
            </button>
            <h2 className={styles.modalTitle}>{selectedBody.name}</h2>
            <div className={styles.modalImageContainer}>
              <img
                src={selectedBody.image_url || '/placeholder-image.jpg'}
                alt={selectedBody.name}
                className={styles.modalImage}
                onError={handleImageError}
              />
            </div>
            <p className={styles.modalDescription}>{selectedBody.description}</p>
            <p className={styles.modalInfo}>
              <strong>Date:</strong> {selectedBody.date || 'Not available'}
            </p>
            <p className={styles.modalInfo}>
              <strong>Keywords:</strong> {selectedBody.keywords || 'Not available'}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

export default CelestialGallery;
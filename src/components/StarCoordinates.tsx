import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import styles from './StarCoordinates.module.css';


interface Coordinates {
  rightAscension: number;
  declination: number;
}

interface CelestialBody {
  name: string;
  coordinates: Coordinates;
  type: string;
  magnitude: number;
  distance: number;
  description: string;
}

const popularStars = [
  { name: 'Sirius', displayName: 'Sirius (Alpha Canis Majoris)' },
  { name: 'Betelgeuse', displayName: 'Betelgeuse (Alpha Orionis)' },
  { name: 'Vega', displayName: 'Vega (Alpha Lyrae)' },
  { name: 'Proxima Centauri', displayName: 'Proxima Centauri' },
  { name: 'Polaris', displayName: 'Polaris (North Star)' },
];

const createStarSystem = (group: THREE.Group, starPosition: THREE.Vector3, starName: string) => {
  const starGeometry = new THREE.SphereGeometry(0.1, 32, 32);
  const starMaterial = new THREE.MeshBasicMaterial({ color: 0xffff00 });
  const starMesh = new THREE.Mesh(starGeometry, starMaterial);
  starMesh.position.copy(starPosition);
  group.add(starMesh);

  const starLight = new THREE.PointLight(0xffff00, 1, 1);
  starLight.position.copy(starPosition);
  group.add(starLight);

  // Create some hypothetical planets
  const planetColors = [0xff0000, 0x00ff00, 0x0000ff, 0xff00ff];
  for (let i = 0; i < 4; i++) {
    const distance = 0.2 + (i * 0.1);
    const angle = Math.random() * Math.PI * 2;
    const planetGeometry = new THREE.SphereGeometry(0.02, 16, 16);
    const planetMaterial = new THREE.MeshBasicMaterial({ color: planetColors[i] });
    const planet = new THREE.Mesh(planetGeometry, planetMaterial);
    planet.position.set(
      starPosition.x + Math.cos(angle) * distance,
      starPosition.y + Math.sin(angle) * distance,
      starPosition.z
    );
    group.add(planet);
  }

  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  if (context) {
    canvas.width = 256;
    canvas.height = 128;
    context.fillStyle = 'white';
    context.font = 'Bold 20px Arial';
    context.fillText(starName, 0, 64);
    const texture = new THREE.CanvasTexture(canvas);
    const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
    const sprite = new THREE.Sprite(spriteMaterial);
    sprite.position.copy(starPosition);
    sprite.position.y += 0.15;
    sprite.scale.set(0.5, 0.25, 1);
    group.add(sprite);
  }
};

const animateCamera = (
  camera: THREE.PerspectiveCamera,
  startPosition: THREE.Vector3,
  endPosition: THREE.Vector3,
  targetPosition: THREE.Vector3,
  duration: number,
  onComplete: () => void
) => {
  const startTime = Date.now();

  function animate() {
    const now = Date.now();
    const progress = Math.min((now - startTime) / duration, 1);
    
    camera.position.lerpVectors(startPosition, endPosition, progress);
    camera.lookAt(targetPosition);

    if (progress < 1) {
      requestAnimationFrame(animate);
    } else {
      onComplete();
    }
  }

  animate();
};

const StarCoordinates: React.FC = () => {
  const [starName, setStarName] = useState<string>('');
  const [celestialBody, setCelestialBody] = useState<CelestialBody | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);

  const addTextLabel = (scene: THREE.Scene, position: THREE.Vector3, text: string) => {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = 256;
    canvas.height = 128;
  
    if (context) {
      context.fillStyle = 'rgba(0, 0, 0, 0)';
      context.fillRect(0, 0, canvas.width, canvas.height);
  
      context.font = 'Bold 24px Arial';
      context.textAlign = 'center';
      context.textBaseline = 'middle';
      context.fillStyle = 'white';
      context.fillText(text, canvas.width / 2, canvas.height / 2);
  
      const texture = new THREE.CanvasTexture(canvas);
      const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
      const sprite = new THREE.Sprite(spriteMaterial);
  
      sprite.position.copy(position);
      sprite.position.multiplyScalar(1.1);
      sprite.scale.set(0.5, 0.25, 1);
  
      scene.add(sprite);
    }
  };

  useEffect(() => {
    if (mountRef.current) {
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
      const renderer = new THREE.WebGLRenderer({ antialias: true });

      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.setClearColor(0x000000, 1);
      mountRef.current.appendChild(renderer.domElement);

      // Add a starfield background
      const starGeometry = new THREE.BufferGeometry();
      const starMaterial = new THREE.PointsMaterial({ color: 0xFFFFFF, size: 0.02 });

      const starVertices = [];
      for (let i = 0; i < 10000; i++) {
        const x = (Math.random() - 0.5) * 2000;
        const y = (Math.random() - 0.5) * 2000;
        const z = (Math.random() - 0.5) * 2000;
        starVertices.push(x, y, z);
      }

      starGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starVertices, 3));
      const stars = new THREE.Points(starGeometry, starMaterial);
      scene.add(stars);

      // Add a celestial sphere
      const sphereGeometry = new THREE.SphereGeometry(10, 32, 32);
      const sphereMaterial = new THREE.MeshBasicMaterial({ 
        color: 0x111111, 
        wireframe: true,
        transparent: true,
        opacity: 0.3
      });
      const celestialSphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
      scene.add(celestialSphere);

      camera.position.z = 15;

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.25;
      controls.enableZoom = true;

      const animate = () => {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      };

      animate();

      sceneRef.current = scene;
      cameraRef.current = camera;
      rendererRef.current = renderer;

      const handleResize = () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      };

      window.addEventListener('resize', handleResize);

      return () => {
        window.removeEventListener('resize', handleResize);
        mountRef.current?.removeChild(renderer.domElement);
        renderer.dispose();
      };
    }
  }, []);

  useEffect(() => {
    if (celestialBody && sceneRef.current && cameraRef.current) {
      const existingSystem = sceneRef.current.getObjectByName('starSystem');
      if (existingSystem) {
        sceneRef.current.remove(existingSystem);
      }

      const starSystemGroup = new THREE.Group();
      starSystemGroup.name = 'starSystem';
      const phi = THREE.MathUtils.degToRad(90 - celestialBody.coordinates.declination);
      const theta = THREE.MathUtils.degToRad(celestialBody.coordinates.rightAscension * 15);
      const starPosition = new THREE.Vector3().setFromSphericalCoords(5, phi, theta);

      createStarSystem(starSystemGroup, starPosition, celestialBody.name);
      sceneRef.current.add(starSystemGroup);
      const startPosition = cameraRef.current.position.clone();
      const endPosition = starPosition.clone().multiplyScalar(1.2);
      const duration = 1000; 

      animateCamera(
        cameraRef.current,
        startPosition,
        endPosition,
        starPosition,
        duration,
        () => {
          console.log('Camera animation completed');
        }
      );
    }
  }, [celestialBody]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!starName.trim()) {
      setError('Please enter a star name');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const response = await axios.get<CelestialBody>(`http://localhost:5001/api/star`, {
        params: { name: starName }
      });
      setCelestialBody(response.data);
    } catch (error) {
      setError('Unable to fetch star data. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handlePopularStarSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const selectedStar = e.target.value;
    if (selectedStar) {
      setStarName(selectedStar);
    }
  };

  return (
    <div className={styles.container}>
      <div ref={mountRef} className={styles.canvas} />
      <div className={styles.overlay}>
        <div className={styles.header}>
          <h2 className={styles.title}>3D Star Viewer</h2>
          <div className={styles.controls}>
            <form onSubmit={handleSubmit} className={styles.form}>
              <input
                type="text"
                value={starName}
                onChange={(e) => setStarName(e.target.value)}
                placeholder="Enter star name or coordinates"
                className={styles.input}
              />
              <button type="submit" disabled={loading} className={styles.button}>
                {loading ? 'Searching...' : 'Lookup'}
              </button>
            </form>
            <select
              onChange={handlePopularStarSelect}
              value=""
              className={styles.select}
            >
              <option value="">Select a popular star</option>
              {popularStars.map(star => (
                <option key={star.name} value={star.name}>{star.displayName}</option>
              ))}
            </select>
          </div>
        </div>
        
        {error && <p className={styles.error}>{error}</p>}
        
        <div className={styles.content}>
          <div className={styles.infoSection}>
            <h3>How to Search</h3>
            <p>You can search for stars using their common names, catalog designations, or coordinates. Here are some examples:</p>
            <ul>
              <li>Common names: Sirius, Betelgeuse, Vega</li>
              <li>Catalog designations: HD 48915 (Sirius), HIP 27989 (Betelgeuse)</li>
              <li>Coordinates: 06 45 08.9 -16 42 58 (Sirius in RA Dec)</li>
            </ul>
          </div>
          
          {celestialBody && (
            <div className={styles.info}>
              <h3>{celestialBody.name}</h3>
              <p><strong>Type:</strong> {celestialBody.type || 'Unknown'}</p>
              <p><strong>Magnitude:</strong> {celestialBody.magnitude?.toFixed(2) || 'Unknown'}</p>
              <p><strong>Distance:</strong> {celestialBody.distance ? `${celestialBody.distance.toFixed(2)} light years` : 'Unknown'}</p>
              <p><strong>Right Ascension:</strong> {celestialBody.coordinates.rightAscension.toFixed(2)} hours</p>
              <p><strong>Declination:</strong> {celestialBody.coordinates.declination.toFixed(2)} degrees</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default StarCoordinates;
import React from 'react';
import { Link } from 'react-router-dom';
import { Card, CardMedia, CardContent, Typography, CardActionArea } from '@mui/material';
import nebulaImage from './nebula.jpeg';
import spaceProject from './space_project.jpg'
import galaxy from './galaxy.jpg';
import pillarsOfCreation from './pillars-of-creation.webp';
import constellation from './constellation.webp';

const FeatureCard: React.FC<{ image: string; title: string; description: string; link: string }> = ({ image, title, description, link }) => (
  <Card className="feature-card" sx={{ maxWidth: 500, maxHeight: 500, backgroundColor: 'teal' }}>
    <CardActionArea component={Link} to={link}>
    <CardMedia 
        component="img" 
        height="140" 
        image={image} 
        alt={title} 
        sx={{ objectFit: 'cover', height: '600px', width: '100%' }}
      />
      <CardContent>
        <Typography gutterBottom variant="h5" component="div">
          {title}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {description}
        </Typography>
      </CardContent>
    </CardActionArea>
  </Card>
);

const Home: React.FC = () => {
  return (
    <div className="relative">
      {/* Background Image */}
      <div className="hero-section">
        <img 
          src={nebulaImage} 
          alt="Space Nebula" 
          className="background-image" 
        />
      </div>

      {/* Content */}
      <div className="overlay">
        {/* Navigation */}
        <nav className="menu">
          <ul className="flex justify-center space-x-4">
            <li><Link to="/" className="text-white hover:text-blue-300">Home</Link></li>
            <li><Link to="/coordinates" className="text-white hover:text-blue-300">Star Coordinates</Link></li>
            <li><Link to="/chatbot" className="text-white hover:text-blue-300">Celestial Chatbot</Link></li>
            <li><Link to="/locations" className="text-white hover:text-blue-300">Stargazing Locations</Link></li>
            <li><Link to="/gallery" className="text-white hover:text-blue-300">Celestial Gallery</Link></li>
          </ul>
        </nav>

        {/* Hero Section */}
        <section className="flex items-center justify-center h-[calc(100vh-64px)] text-white">
          <div className="text-center">
            <h1 className="text-5xl font-bold mb-4">Explore the Cosmos</h1>
            <p className="text-xl mb-8">Embark on a journey through the stars with Celestial Explorer and Discover the Universe</p>
            {/* <Link
              // to="/chatbot" 
              className="text-white hover:text-blue-300"
            >
              Start Your Cosmic Journey
            </Link> */}
          </div>
        </section>

        {/* Features Section */}
        <h2 className="text-5xl font-bold mb-8 text-center text-white">Discover the Universe</h2>
        <section className="features-section py-16">
          <div className="flex justify-center">
            <FeatureCard 
              image={constellation} 
              title="Star Coordinates" 
              description="Navigate the night sky with precision" 
              link="/coordinates" 
            />
            <FeatureCard 
              image={spaceProject} 
              title="Celestial Chatbot" 
              description="Converse with our AI about cosmic wonders" 
              link="/chatbot" 
            />
            <FeatureCard 
              image={galaxy}
              title="Stargazing Locations" 
              description="Find the best spots for celestial observation" 
              link="/locations" 
            />
            <FeatureCard 
              image={pillarsOfCreation}
              title="Celestial Gallery" 
              description="Witness breathtaking images of the cosmos" 
              link="/gallery" 
            />
          </div>
        </section>
      </div>
    </div>
  );
};

export default Home;

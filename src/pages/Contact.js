import React, { useState, useEffect } from 'react';

// Responsive: custom hook for screen size
function useWindowWidth() {
  const [width, setWidth] = useState(window.innerWidth);
  useEffect(() => {
    const handleResize = () => setWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  return width;
}

function Contact() {
  // Array of image paths
  const images = [
    '/contact_us_bg.jpg',
    '/contact1.png',
    '/contact2.png'
  ];
  const [imgIdx, setImgIdx] = useState(0);

  // Responsive setup
  const width = useWindowWidth();
  const isMobile = width < 700;

  // Cycle image every 3 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setImgIdx(prev => (prev + 1) % images.length);
    }, 3000);
    return () => clearInterval(interval);
  }, [images.length]);

  // Responsive styles
  const styles = {
    page: {
      backgroundColor: '#111',
      color: '#fff',
      minHeight: '100vh',
      padding: isMobile ? '6px 2vw' : '10px 12px',
      fontFamily: "'Segoe UI', Arial, sans-serif",
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    },
    container: {
      width: '100%',
      maxWidth: isMobile ? '98vw' : '850px',
      background: '#191c24',
      borderRadius: '18px',
      marginTop: '10px',
      boxShadow: '0 6px 36px #1e90ff40',
      display: 'flex',
      flexDirection: isMobile ? 'column' : 'row',
      overflow: 'hidden',
    },
    left: {
      flex: isMobile ? 'unset' : 1,
      minWidth: 0,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: isMobile ? '#161a29' : 'linear-gradient(135deg, #0e1831 60%, #1e90ff 120%)',
      padding: isMobile ? '16px 0 0 0' : 0,
    },
    image: {
      width: isMobile ? '100%' : '360px',
      maxWidth: '360px',
      height: isMobile ? 'auto' : '280px',
      objectFit: 'cover',
      display: 'block',
      borderRadius: isMobile ? '18px 18px 0 0' : '0 0 0 18px',
      margin: isMobile ? '0 0 20px 0' : '40px 0',
      boxShadow: '0 0 30px #1e90ff22',
      transition: 'opacity 0.6s',
    },
    right: {
      flex: isMobile ? 'unset' : 1.2,
      padding: isMobile ? '22px 6vw' : '40px 32px',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      alignItems: isMobile ? 'center' : 'flex-start',
    },
    heading: {
      fontSize: isMobile ? '1.45rem' : '2.35rem',
      fontWeight: 700,
      color: '#1e90ff',
      marginBottom: '18px',
      textTransform: 'uppercase',
      textShadow: '0 0 16px #1e90ff44',
      letterSpacing: '2px',
      textAlign: isMobile ? 'center' : 'left',
    },
    button: {
      display: 'inline-block',
      padding: isMobile ? '12px 28px' : '15px 44px',
      fontSize: isMobile ? '1.01rem' : '1.18rem',
      fontWeight: 600,
      color: '#fff',
      backgroundColor: '#1e90ff',
      border: 'none',
      borderRadius: '8px',
      textDecoration: 'none',
      boxShadow: '0 2px 12px #1e90ff35',
      cursor: 'pointer',
      margin: '24px 0 18px 0',
      transition: 'background 0.2s',
      letterSpacing: '1px',
      textAlign: 'center',
    },
    smallLabel: {
      fontSize: isMobile ? '0.95rem' : '0.98rem',
      color: '#44bedf',
      margin: 0,
      marginBottom: '3px',
      fontWeight: 500,
      letterSpacing: '1px',
      textAlign: isMobile ? 'center' : 'left',
    },
    infoRow: {
      fontSize: isMobile ? '1rem' : '1.14rem',
      margin: '8px 0 0 0',
      textAlign: isMobile ? 'center' : 'left',
      wordBreak: 'break-word',
    },
    link: {
      color: '#1e90ff',
      textDecoration: 'none',
      fontWeight: 500,
      marginLeft: '5px',
    },
    caption: {
      marginTop: isMobile ? '22px' : '35px',
      color: '#bbbbbb',
      fontSize: isMobile ? '0.97rem' : '1.01rem',
      fontWeight: 400,
      letterSpacing: '0.5px',
      textAlign: 'center',
    },
    footer: {
      textAlign: 'center',
      color: '#bbb',
      fontSize: isMobile ? '0.95rem' : '1.05rem',
      marginTop: '28px',
      letterSpacing: '0.8px',
    },
  };

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        {/* Left: Rotating Image */}
        <div style={styles.left}>
          <img
            src={images[imgIdx]}
            alt="Contact"
            style={styles.image}
            loading="lazy"
          />
        </div>

        {/* Right: Content */}
        <div style={styles.right}>
          <h1 style={styles.heading}>Contact Us</h1>
          {/* Send Email to all three addresses */}
          <a
            href="mailto:deepesh.kumar.ug22@nsut.ac.in,prateek.dhanker.ug22@nsut.ac.in,gaurav.kumar.ug22@nsut.ac.in"
            style={styles.button}
            title="Send your query to all team members"
          >
            📧 Send Email
          </a>

          <div>
            {/* Emails */}
            <p style={styles.smallLabel}>Email</p>
            <div style={styles.infoRow}>
              <a href="mailto:deepesh.kumar.ug22@nsut.ac.in" style={styles.link}>
                deepesh.kumar.ug22@nsut.ac.in
              </a>
            </div>
            <div style={styles.infoRow}>
              <a href="mailto:prateek.dhanker.ug22@nsut.ac.in" style={styles.link}>
                prateek.dhanker.ug22@nsut.ac.in
              </a>
            </div>
            <div style={styles.infoRow}>
              <a href="mailto:gaurav.kumar.ug22@nsut.ac.in" style={styles.link}>
                gaurav.kumar.ug22@nsut.ac.in
              </a>
            </div>

            {/* Phone Numbers */}
            <p style={styles.smallLabel}>Phone</p>
            <div style={styles.infoRow}>
              <a href="tel:+917982460774" style={styles.link}>
                +91-79824 60774
              </a>
            </div>
            <div style={styles.infoRow}>
              <a href="tel:+917988898595" style={styles.link}>
                +91-79888 98595
              </a>
            </div>
            <div style={styles.infoRow}>
              <a href="tel:+919599871719" style={styles.link}>
                +91-95998 71719
              </a>
            </div>
          </div>

          <div style={styles.caption}>
            We're here to help! For any technical queries, project collaborations, or support with crowd management solutions, connect directly using the above information.
          </div>
          <div style={styles.footer}>
            © 2025 VigilNet Project Team
          </div>
        </div>
      </div>
    </div>
  );
}

export default Contact;

import React, { useState, useEffect } from 'react';

function useWindowWidth() {
  const [width, setWidth] = useState(window.innerWidth);

  useEffect(() => {
    const handleResize = () => setWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return width;
}

function FAQ() {
  const width = useWindowWidth();
  const isMobile = width < 700;

  const styles = {

    page: {
      backgroundColor: '#111',
      color: '#fff',
      minHeight: '100vh',
      padding: isMobile ? '22px 5px' : '40px 16px',
      fontFamily: "'Segoe UI', Arial, sans-serif",
      textAlign: 'center',
    },
    flexWrap: {
      display: 'flex',
      alignItems: isMobile ? 'center' : 'flex-start',
      justifyContent: 'center',
      maxWidth: '1100px',
      margin: '0 auto',
      gap: isMobile ? '16px' : '36px',
      flexDirection: isMobile ? 'column' : 'row',
      flexWrap: 'wrap',
      width: '100%',
    },
    imageBox: {
      flex: isMobile ? 'unset' : '0 0 310px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      width: isMobile ? '100%' : '310px',
      marginBottom: isMobile ? '15px' : '0',
    },
    faqImage: {
      width: isMobile ? '75%' : '95%',
      marginTop: isMobile ? '18px' : '40px',
      maxWidth: '270px',
      borderRadius: '15px',
      boxShadow: '0 4px 22px #1e90ff30',
    },
    contentBox: {
      flex: 1,
      minWidth: isMobile ? '0' : '250px',
      width: '100%',
    },
    heading: {
      fontSize: isMobile ? '1.35rem' : '2.3rem',
      fontWeight: 'bold',
      color: '#1e90ff',
      marginBottom: '18px',
      textTransform: 'uppercase',
      letterSpacing: '1.5px',
      textShadow: '0 0 9px rgba(30,144,255,0.6)',
      textAlign: 'left',
    },
    subtext: {
      fontSize: isMobile ? '1rem' : '1.12rem',
      color: '#ccc',
      maxWidth: '650px',
      margin: '0 0 36px 0',
      lineHeight: '1.7',
      textAlign: 'left',
    },
    faqList: {
      maxWidth: isMobile ? '96vw' : '700px',
      margin: '0 auto',
      textAlign: 'left',
    },
    faqItem: {
      backgroundColor: 'rgba(30,144,255,0.10)',
      border: '1px solid #1e90ff',
      borderRadius: '10px',
      padding: isMobile ? '13px' : '20px',
      marginBottom: isMobile ? '11px' : '18px',
      transition: 'transform 0.3s, box-shadow 0.3s',
      cursor: 'pointer',
      boxShadow: '0 2px 8px #1e90ff22',
    },
    question: {
      fontSize: isMobile ? '1rem' : '1.17rem',
      fontWeight: 'bold',
      marginBottom: '10px',
      color: '#1e90ff',
      letterSpacing: '0.5px',
    },
    answer: {
      fontSize: isMobile ? '0.95rem' : '1.03rem',
      color: '#ddd',
      lineHeight: '1.6',
    },
  };

  const handleMouseEnter = (e) => {
    e.currentTarget.style.transform = 'translateY(-3px)';
    e.currentTarget.style.boxShadow = '0 10px 28px rgba(30,144,255,0.22)';
  };

  const handleMouseLeave = (e) => {
    e.currentTarget.style.transform = 'translateY(0)';
    e.currentTarget.style.boxShadow = '0 2px 8px #1e90ff22';
  };

  const faqs = [
    {
      q: '1. What is VigilNet Crowd Management?',
      a: 'VigilNet is a real-time crowd monitoring and alerting system that uses AI-powered analytics to detect anomalies and provide actionable insights.',
    },
    {
      q: '2. How do I integrate my cameras or NVR systems?',
      a: 'You can integrate RTSP streams, NVRs, or cloud-based video sources directly through the VigilNet dashboard with minimal configuration.',
    },
    {
      q: '3. Does VigilNet support multi-camera and multi-location monitoring?',
      a: 'Yes, VigilNet can handle multiple cameras across different sites simultaneously, making it suitable for large-scale events or smart city deployments.',
    },
    {
      q: '4. What research areas does this project contribute to?',
      a: 'VigilNet contributes to fields such as computer vision, deep learning for crowd analysis, anomaly detection algorithms, and edge computing for real-time video analytics.',
    },
    {
      q: '5. Which AI models are commonly used in VigilNet for crowd analysis?',
      a: 'Techniques like YOLO for object detection and CSRNet for crowd density estimation are used to achieve high accuracy in real-time crowd monitoring.',
    },
    {
      q: '6. How does VigilNet handle privacy and compliance?',
      a: 'The system can anonymize faces and complies with privacy standards such as GDPR by processing only metadata or anonymized frames when required.',
    },
    {
      q: '7. Can VigilNet be extended for academic or research purposes?',
      a: 'Yes, VigilNet supports REST APIs and modular architecture, making it easy to integrate with experimental models or data pipelines for academic research.',
    },
    {
      q: '8. What technologies power VigilNet’s backend?',
      a: 'The backend leverages Node.js and Python-based AI services, supported by WebSockets for real-time alerts and a scalable cloud infrastructure.',
    },
    {
      q: '9. How can VigilNet assist in emergency planning or event management research?',
      a: 'By providing real-time crowd density heatmaps and anomaly alerts, VigilNet helps researchers analyze evacuation strategies, event safety, and urban planning.',
    },
    {
      q: '10. Does VigilNet work in low-light or adverse weather conditions?',
      a: 'Yes, VigilNet uses pre-processing techniques like low-light enhancement and noise reduction to maintain detection accuracy even in challenging environments.',
    },
  ];

  return (
    <div style={styles.page}>
      <div style={styles.flexWrap}>
        {/* Left-side FAQ image */}
        <div style={styles.imageBox}>
          <img src="/faq.jpg" alt="FAQ illustration" style={styles.faqImage} />
        </div>
        {/* Right-side FAQ content */}
        <div style={styles.contentBox}>
          <h1 style={styles.heading}>Frequently Asked Questions</h1>
          <p style={styles.subtext}>
            Here you'll find answers to common and research-level questions about the VigilNet Crowd Management System.
          </p>
          <div style={styles.faqList}>
            {faqs.map((faq, index) => (
              <div
                key={index}
                style={styles.faqItem}
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
              >
                <div style={styles.question}>{faq.q}</div>
                <div style={styles.answer}>{faq.a}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default FAQ;

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';






const securityQuestions = [
  "What is your pet's name?",
  "What is your mother's maiden name?",
  "What is your favorite color?",
  "What city were you born in?",
  "What was your first school name?"
];



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

function Signup() {
  const [role, setRole] = useState('user');
  const [showPopup, setShowPopup] = useState(false);
  const navigate = useNavigate(); // Add navigation hook
  const [apiMessage, setApiMessage] = useState('');
  const [showMessage, setShowMessage] = useState(false);
  const [isError, setIsError] = useState(false);
  const width = useWindowWidth();
  const isMobile = width < 700;



  const styles = {
    page: {
      minHeight: '100vh',
      width: '100vw',
      background: 'url(/signup_bg.jpg) no-repeat center center fixed',
      backgroundSize: 'cover',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: isMobile ? 'flex-start' : 'center',  // TOP on mobile, center on desktop
      fontFamily: "'Segoe UI', 'Poppins', Arial, sans-serif",
      animation: 'fadeInPage 1s ease forwards',
      padding: isMobile ? '20px 0' : undefined, // Add vertical padding on mobile to prevent cut-off
      overflowY: 'auto'
    },
    container: {
      display: 'flex',
      background: 'white',
      borderRadius: '24px',
      boxShadow: '0 5px 36px rgba(217, 219, 222, 0.1), 0 1.5px 12px #d8dbdd55',
      overflow: 'hidden',
      maxWidth: isMobile ? '99vw' : '920px',
      width: '100%',
      height: isMobile ? 'auto' : '90vh',
      minHeight: isMobile ? '100vh' : '500px',
      flexDirection: isMobile ? 'column' : 'row',
      animation: 'scaleUp 0.6s cubic-bezier(0.4, 0, 0.2, 1) forwards',
      transformOrigin: 'center center',
      height: isMobile ? 'auto' : '90vh',
      minHeight: isMobile ? 'unset' : '500px',
      marginTop: isMobile ? '0' : undefined,
      marginBottom: isMobile ? '30px' : undefined,
    },
    left: {
      flex: 1.2,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      height: isMobile ? '160px' : '100%',
      background: '#000',
      animation: 'slideInLeft 0.8s ease forwards',
      padding: isMobile ? '18px 0 0 0' : 0,
    },
    leftImg: {
      maxHeight: isMobile ? '120px' : '90%',
      maxWidth: '100%',
      objectFit: 'contain',
      borderRadius: '14px',
      filter: 'drop-shadow(0 4px 6px rgba(0,0,0,0.3))',
      marginTop: isMobile ? '0' : '16px',
      marginBottom: isMobile ? '0' : '0'
    },
    right: {
      flex: 1.1,
      display: 'flex',
      flexDirection: 'column',
      justifyContent: isMobile ? 'flex-start' : 'center',
      background: '#000',
      borderRadius: isMobile ? '0 0 18px 18px' : '0 18px 18px 0',
      minWidth: isMobile ? '0' : '360px',
      height: isMobile ? 'auto' : '100%',
      overflowY: 'auto',
      padding: isMobile ? '32px 7vw 28px 7vw' : '38px clamp(18px, 5vw, 50px)',
      animation: 'slideInRight 0.8s ease forwards',
    },
    heading: {
      fontSize: isMobile ? '1.35rem' : '2.1rem',
      fontWeight: 700,
      color: '#1e90ff',
      marginBottom: '17px',
      marginTop: isMobile ? '42px' : '890px',           // Nice top gap on mobile, none on desktop
      textAlign: isMobile ? 'center' : 'left',
      letterSpacing: '1.6px',
      textShadow: '1px 1px 3px rgba(0,0,0,0.3)'
    },

    label: {
      color: '#1e90ff',
      fontWeight: 500,
      marginBottom: 2,
      fontSize: isMobile ? '0.98rem' : '1rem'
    },
    input: {
      width: '100%',
      padding: isMobile ? '9px 8px' : '10px 14px',
      marginBottom: '14px',
      borderRadius: '9px',
      border: '1.4px solid #1e90ff',
      fontSize: isMobile ? '0.96rem' : '1rem',
      background: '#fbfcfcff',
      fontWeight: 500,
      color: '#000000ff',
      transition: 'border-color 0.3s ease',
    },
    select: {
      width: '100%',
      padding: isMobile ? '9px 8px' : '10px 14px',
      marginBottom: '14px',
      borderRadius: '9px',
      border: '1.4px solid #1e90ff',
      fontSize: isMobile ? '0.96rem' : '1rem',
      background: '#fbfcfcff',
      fontWeight: 500,
      color: '#000000ff',
      transition: 'border-color 0.3s ease',
    },
    submit: {
      width: '100%',
      padding: isMobile ? '11px 0' : '13px 0',
      borderRadius: '8px',
      background: 'linear-gradient(90deg, #4c90eeff 60%, #0248d3ff 100%)',
      color: '#fff',
      fontWeight: 700,
      fontSize: isMobile ? '1.01rem' : '1.15rem',
      border: 'none',
      marginTop: '15px',
      marginBottom: isMobile ? '38px' : '100px',
      boxShadow: '0 2px 14px #eaeaea45',
      letterSpacing: '1.1px',
      cursor: 'pointer',
      transition: 'transform 0.25s ease',
    },
    toggleWrap: {
      display: 'flex',
      alignItems: 'center',
      gap: '15px',
      margin: '16px 0 5px 0',
      animation: 'fadeIn 1.2s ease forwards',
      flexDirection: isMobile ? 'column' : 'row',
      justifyContent: isMobile ? 'center' : undefined,
    },
    toggleBtn: (active) => ({
      padding: isMobile ? '7px 16vw' : '8px 26px',
      borderRadius: '16px',
      border: '2px solid #1e90ff',
      background: active ? '#1e90ff' : '#eef6faff',
      color: active ? '#fff' : '#1e90ff',
      fontWeight: 600,
      cursor: 'pointer',
      boxShadow: active ? '0 2.5px 12px #a4a3a0ab' : undefined,
      fontSize: isMobile ? '0.96rem' : '1rem',
      letterSpacing: '0.8px',
      transition: 'background-color 0.3s ease, color 0.3s ease, box-shadow 0.3s ease',
    }),
    popup: {
      position: 'fixed',
      top: '20px',
      right: isMobile ? '5vw' : '20px',
      background: 'linear-gradient(90deg, #4c90eeff 60%, #0248d3ff 100%)',
      color: '#fff',
      padding: '14px 24px',
      borderRadius: '10px',
      fontSize: '1rem',
      fontWeight: 600,
      boxShadow: '0 3px 16px rgba(0,0,0,0.2)',
      zIndex: 1000,
      animation: 'fadeIn 0.5s ease',
      maxWidth: isMobile ? '86vw' : '350px',
      textAlign: 'center'
    },

    '@keyframes fadeIn': {
      from: { opacity: 0 },
      to: { opacity: 1 },
    },
    '@keyframes fadeInPage': {
      from: { opacity: 0, transform: 'translateY(10px)' },
      to: { opacity: 1, transform: 'translateY(0)' },
    },
    '@keyframes scaleUp': {
      from: { opacity: 0, transform: 'scale(0.95)' },
      to: { opacity: 1, transform: 'scale(1)' },
    },
    '@keyframes slideInLeft': {
      from: { opacity: 0, transform: 'translateX(-40px)' },
      to: { opacity: 1, transform: 'translateX(0)' },
    },
    '@keyframes slideInRight': {
      from: { opacity: 0, transform: 'translateX(40px)' },
      to: { opacity: 1, transform: 'translateX(0)' },
    }
  };




  const [form, setForm] = useState({
    firstName: '', middleName: '', lastName: '',
    dob: '', gender: 'Male', country: 'India', email: '',
    password: '', confirmPassword: '', mobile: '', altMobile: '',
    deptName: '', deptId: '', address: '', state: '', aadhar: '', securityQuestion: '', securityAnswer: ''
  });

  const onChange = e => setForm(f => ({ ...f, [e.target.name]: e.target.value }));

  // const handleSubmit = async (e) => {
  //   e.preventDefault();
  //   // Validate passwords match
  //   if (form.password !== form.confirmPassword) {
  //     alert('Passwords do not match!');
  //     return;
  //   }
  //   try {
  //     const response = await fetch('http://192.168.1.34:5000/api/signup', {
  //       method: 'POST',
  //       headers: { 'Content-Type': 'application/json' },
  //       body: JSON.stringify({
  //         ...form,
  //         role,
  //       }),
  //     });
  //     if (!response.ok) {
  //       throw new Error(`HTTP error! status: ${response.status}`);
  //     }
  //     const result = await response.json();
  //     setApiMessage(result.message);
  //     setIsError(!result.success);
  //     setShowMessage(true);

  //     if (result.success) {
  //       setShowPopup(true);
  //       setTimeout(() => {
  //         setShowPopup(false);
  //         navigate('/login');
  //       }, 2000);
  //     } else {
  //       alert(result.message || 'Signup failed. Please try again.');
  //     }

  //     setTimeout(() => setShowMessage(false), 4000);

  //   } catch (err) {
  //     console.error('Signup error:', err);
  //     setApiMessage('Unable to connect to server');
  //     setIsError(true);
  //     setShowMessage(true);
  //     setTimeout(() => setShowMessage(false), 4000);
  //     // alert('Server connection error. Please check your connection and try again.');
  //   }
  // };  

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (form.password !== form.confirmPassword) {
      alert('Passwords do not match!');
      return;
    }

    try {
      const response = await fetch('http://192.168.1.34:5000/api/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...form, role }),
      });

      // Try to parse JSON response, even if it's a 400 or 500
      let result;
      try {
        result = await response.json();
      } catch {
        result = { success: false, message: 'Invalid server response' };
      }

      console.log('Backend response:', result); // <-- helpful for debugging

      setApiMessage(result.message || 'Unknown error occurred');
      setIsError(!result.success);
      setShowMessage(true);

      if (response.ok && result.success) {
        setShowPopup(true);
        setTimeout(() => {
          setShowPopup(false);
          navigate('/login');
        }, 2000);
      }

      setTimeout(() => setShowMessage(false), 4000);

    } catch (err) {
      console.error('Signup error (network):', err);
      setApiMessage('Unable to connect to server');
      setIsError(true);
      setShowMessage(true);
      setTimeout(() => setShowMessage(false), 4000);
    }
  };



  return (
    <div style={styles.page}>
      {showPopup && <div style={styles.popup}>Signup successful!</div>}
      {showMessage && (
        <div style={{
          position: 'fixed',
          bottom: '24px',
          left: '50%',
          transform: 'translateX(-50%)',
          backgroundColor: isError ? '#e74c3c' : '#2ecc71',
          color: 'white',
          padding: '14px 28px',
          borderRadius: '24px',
          boxShadow: '0 6px 12px rgba(0,0,0,0.2)',
          fontWeight: '600',
          zIndex: 9999,
          textAlign: 'center',
          minWidth: '280px'
        }}>
          {apiMessage}
        </div>
      )}

      <div style={styles.container}>
        {/* Left illustration */}
        <div style={styles.left}>
          <img src="/signup.jpg" alt="Signup Left" style={styles.leftImg} />
        </div>

        {/* Right-side Signup form */}
        <div style={styles.right}>
          <div style={styles.heading}>Sign Up</div>
          <div style={styles.toggleWrap}>
            <button type="button" style={styles.toggleBtn(role === 'admin')} onClick={() => setRole('admin')}>Admin</button>
            <button type="button" style={styles.toggleBtn(role === 'user')} onClick={() => setRole('user')}>Normal User</button>
          </div>
          <form onSubmit={handleSubmit}>
            <div>
              <label style={styles.label}>First Name</label>
              <input name="firstName" style={styles.input} value={form.firstName} onChange={onChange} required />
            </div>
            <div>
              <label style={styles.label}>Middle Name</label>
              <input name="middleName" style={styles.input} value={form.middleName} onChange={onChange} />
            </div>
            <div>
              <label style={styles.label}>Last Name</label>
              <input name="lastName" style={styles.input} value={form.lastName} onChange={onChange} required />
            </div>
            <div>
              <label style={styles.label}>Date of Birth</label>
              <input name="dob" type="date" style={styles.input} value={form.dob} onChange={onChange} required />
            </div>
            <div>
              <label style={styles.label}>Gender</label>
              <select name="gender" style={styles.select} value={form.gender} onChange={onChange} required>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Transgender">Transgender</option>
              </select>
            </div>
            {role === "admin" ?
              <>
                <div>
                  <label style={styles.label}>Department Name</label>
                  <input name="deptName" style={styles.input} value={form.deptName} onChange={onChange} required />
                </div>
                <div>
                  <label style={styles.label}>Department ID</label>
                  <input name="deptId" style={styles.input} value={form.deptId} onChange={onChange} required />
                </div>
              </>
              :
              <div>
                <label style={styles.label}>Aadhar Card</label>
                <input name="aadhar" style={styles.input} value={form.aadhar} onChange={onChange} required />
              </div>
            }
            <div>
              <label style={styles.label}>Address</label>
              <input name="address" style={styles.input} value={form.address} onChange={onChange} required />
            </div>
            <div>
              <label style={styles.label}>State</label>
              <input name="state" style={styles.input} value={form.state} onChange={onChange} required />
            </div>
            <div>
              <label style={styles.label}>Country</label>
              <input name="country" style={styles.input} value={form.country} onChange={onChange} required />
            </div>
            <div>
              <label style={styles.label}>Email</label>
              <input name="email" type="email" style={styles.input} value={form.email} onChange={onChange} required />
            </div>
            <div>
              <label style={styles.label}>Password</label>
              <input name="password" type="password" style={styles.input} value={form.password} onChange={onChange} required />
            </div>
            <div>
              <label style={styles.label}>Confirm Password</label>
              <input name="confirmPassword" type="password" style={styles.input} value={form.confirmPassword} onChange={onChange} required />
            </div>
            <div>
              <label style={styles.label}>Mobile Number</label>
              <input name="mobile" style={styles.input} value={form.mobile} onChange={onChange} required />
            </div>
            <div>
              <label style={styles.label}>Alternate Mobile Number</label>
              <input name="altMobile" style={styles.input} value={form.altMobile} onChange={onChange} />
            </div>
            <div>
              <label style={styles.label}>Security Question</label>
              <select name="securityQuestion" style={styles.select} value={form.securityQuestion} onChange={onChange} required>
                <option value="">Select a question</option>
                {securityQuestions.map(q => <option key={q}>{q}</option>)}
              </select>
            </div>
            <div>
              <label style={styles.label}>Answer</label>
              <input name="securityAnswer" style={styles.input} value={form.securityAnswer} onChange={onChange} required />
            </div>
            <button type="submit" style={styles.submit}>Sign Up</button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default Signup;

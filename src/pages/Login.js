import { Link } from 'react-router-dom';
import React, { useState } from "react";
import { FaUser, FaLock } from "react-icons/fa";
import { useNavigate } from 'react-router-dom';

// const styles = {
//   page: { minHeight: '100vh', width: '100vw', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: "'Poppins', 'Segoe UI', Arial, sans-serif",background: "url('/login_bg.jpg') center/cover no-repeat",display: "flex", },
//   container: { display: 'flex', maxWidth: '900px', width: '100%', minHeight: '480px', background: 'white', borderRadius: '24px', boxShadow: '0 7px 32px #0aa4fd33', overflow: 'hidden' },
//   left: { flex: "1 0 330px", background: "#f0f7ff", display: "flex", alignItems: "center", justifyContent: "center" },
//   leftImg: { width: "98%", maxWidth: "340px", maxHeight: "75vh", objectFit: "contain", borderRadius: "18px 0 0 18px" },
//   right: { flex: "1.1", padding: '30px 40px', display: 'flex', flexDirection: "column", justifyContent: "center" },
//   headerRow: { display: "flex", alignItems: "center", marginBottom: 30 },
//   logoImg: { height: 40, marginRight: 12 },
//   loginTitle: { fontSize: 22, fontWeight: 700, color: '#0288d1' },
//   toggleWrap: { display: "flex", gap: "18px", marginBottom: "25px" },
//   toggleBtn: (active) => ({
//     cursor: "pointer",
//     padding: "10px 32px",
//     borderRadius: "8px",
//     fontWeight: 600,
//     color: active ? "#fff" : "#0288d1",
//     background: active ? "linear-gradient(90deg,#0288d1,#0d47a1)" : "transparent",
//     border: active ? "none" : "2px solid #0288d1"
//   }),
//   inputWrap: { display: "flex", alignItems: "center", gap: "14px", marginBottom: 18, border: "1.6px solid #0288d1", borderRadius: "8px", padding: 8 },
//   input: { border: "none", outline: "none", flex: 1, fontSize: 16, fontWeight: 600, backgroundColor: 'transparent' },
//   button: { marginTop: 18, background: "linear-gradient(90deg,#0288d1,#0d47a1)", color: "#fff", cursor: "pointer", borderRadius: 8, fontWeight: 700, fontSize: "1.1rem", padding: "12px", border: "none" },
//   forgot: { color: "#0288d1", cursor: "pointer", marginTop: 12, fontWeight: 600, userSelect: "none" },
//   welcome: { marginTop: 44, fontWeight: 700, fontSize: "1.6rem", textAlign: "center", color: "#555" },
//   signupWrap: { marginTop: 13, fontWeight: 600, fontSize: "1rem", textAlign: "center", color: "#777" },
//   signup: { marginTop: 12, padding: "10px 30px", borderRadius: 8, background: "linear-gradient(90deg,#0288d1,#0d47a1)", color: "#fff", fontWeight: 700, fontSize: "1.05rem", border: "none", cursor: "pointer" },
//   /* Forgot Modal styles */
//   modalOverlay: { position: "fixed", inset: 0, backgroundColor: "rgba(0, 0, 0, 0.35)", display: "flex", justifyContent: "center", alignItems: "center", zIndex: 9999 },
//   modal: { backgroundColor: "#f9f9f9", padding: 24, borderRadius: 12, maxWidth: 380, width: "100%", textAlign: "center", boxShadow: "0 0 24px #00000044" },
//   modalTitle: { fontWeight: 700, fontSize: "1.2rem", marginBottom: 20, color: "#0d47a1" },
//   modalInput: { width: "100%", padding: 10, marginBottom: 12, fontSize: 16, borderRadius: 6, border: "1.6px solid #0288d1", outline: "none" },
//   modalBtn: { width: "100%", padding: 12, background: "linear-gradient(90deg,#0288d1,#0d47a1)", color: "white", fontWeight: 700, fontSize: "1rem", border: "none", borderRadius: 6, cursor: "pointer", marginTop: 8 },
//   cancelBtn: { marginTop: 10, cursor: "pointer", color: "#0d47a1", fontWeight: 600, border: "none", background: "transparent" },
// };




function useWindowWidth() {
  const [width, setWidth] = useState(window.innerWidth);
  React.useEffect(() => {
    const handleResize = () => setWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  return width;
}


function Login() {
  const [role, setRole] = useState('user');
  const [formData, setFormData] = useState({ username: "", password: "" });
  const [showForgot, setShowForgot] = useState(false);


  const width = useWindowWidth();
  const isMobile = width < 700;

  const styles = {
    page: {
      minHeight: '100vh',
      width: '100vw',
      display: 'flex',
      alignItems: 'center',
      justifyContent: isMobile ? 'flex-start' : 'center', // top align on mobile
      fontFamily: "'Poppins', 'Segoe UI', Arial, sans-serif",
      background: "url('/login_bg.jpg') center/cover no-repeat",
      padding: isMobile ? '20px 0' : undefined,
      overflowY: "auto",
    },
    container: {
      display: 'flex',
      maxWidth: isMobile ? '99vw' : '900px',
      width: '100%',
      minHeight: isMobile ? '80vh' : '480px',
      background: 'white',
      borderRadius: '24px',
      boxShadow: '0 7px 32px #0aa4fd33',
      overflow: 'hidden',
      flexDirection: isMobile ? 'column' : 'row',
      marginTop: isMobile ? '0' : undefined,
      marginBottom: isMobile ? '24px' : undefined,
    },
    left: {
      flex: isMobile ? 'unset' : "1 0 330px",
      background: "#f0f7ff",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      padding: isMobile ? '30px 0 0 0' : 0,
      minHeight: isMobile ? "130px" : "inherit",
    },
    leftImg: {
      width: isMobile ? "95vw" : "98%",
      maxWidth: "340px",
      maxHeight: isMobile ? "110px" : "75vh",
      objectFit: "contain",
      borderRadius: isMobile ? "18px 18px 0 0" : "18px 0 0 18px",
    },
    right: {
      flex: isMobile ? 'unset' : "1.1",
      padding: isMobile ? '24px 7vw 24px 7vw' : '30px 40px',
      display: 'flex',
      flexDirection: "column",
      justifyContent: isMobile ? "flex-start" : "center",
      alignItems: "center",
    },
    headerRow: {
      display: "flex",
      alignItems: "center",
      marginBottom: isMobile ? 24 : 30,
      justifyContent: isMobile ? "center" : undefined
    },
    logoImg: {
      height: isMobile ? 32 : 40,
      marginRight: 12,
    },
    loginTitle: {
      fontSize: isMobile ? '1.01rem' : 22,
      fontWeight: 700,
      color: '#0288d1',
      textAlign: isMobile ? "center" : "left",
    },
    toggleWrap: {
      display: "flex",
      gap: isMobile ? "9px" : "18px",
      marginBottom: "20px",
      flexDirection: isMobile ? "column" : "row",
      justifyContent: isMobile ? "center" : undefined,
      alignItems: "center",
    },
    toggleBtn: (active) => ({
      cursor: "pointer",
      padding: isMobile ? "8px 24vw" : "10px 32px",
      borderRadius: "8px",
      fontWeight: 600,
      color: active ? "#fff" : "#0288d1",
      background: active ? "linear-gradient(90deg,#0288d1,#0d47a1)" : "transparent",
      border: active ? "none" : "2px solid #0288d1"
    }),
    inputWrap: {
      display: "flex",
      alignItems: "center",
      gap: isMobile ? "9px" : "14px",
      marginBottom: isMobile ? 14 : 18,
      border: "1.6px solid #0288d1",
      borderRadius: "8px",
      padding: isMobile ? 6 : 8,
      background: "#f8fcff",
      width: isMobile ? "94vw" : "100%",
      maxWidth: "350px"
    },
    input: {
      border: "none",
      outline: "none",
      flex: 1,
      fontSize: isMobile ? 15 : 16,
      fontWeight: 600,
      backgroundColor: 'transparent',
      minWidth: 0
    },
    button: {
      marginTop: isMobile ? 14 : 18,
      background: "linear-gradient(90deg,#0288d1,#0d47a1)",
      color: "#fff",
      cursor: "pointer",
      borderRadius: 8,
      fontWeight: 700,
      fontSize: isMobile ? "1rem" : "1.1rem",
      padding: isMobile ? "11px" : "12px",
      border: "none",
      width: isMobile ? "94vw" : "100%",
      maxWidth: "350px"
    },
    forgot: { color: "#0288d1", cursor: "pointer", marginTop: 12, fontWeight: 600, userSelect: "none", fontSize: isMobile ? "1rem" : undefined },
    welcome: { marginTop: isMobile ? 26 : 44, fontWeight: 700, fontSize: isMobile ? "1.18rem" : "1.6rem", textAlign: "center", color: "#555" },
    signupWrap: { marginTop: 10, fontWeight: 600, fontSize: isMobile ? "0.95rem" : "1rem", textAlign: "center", color: "#777" },
    signup: { marginTop: 10, padding: isMobile ? "11px 22vw" : "10px 30px", borderRadius: 8, background: "linear-gradient(90deg,#0288d1,#0d47a1)", color: "#fff", fontWeight: 700, fontSize: isMobile ? "1rem" : "1.05rem", border: "none", cursor: "pointer", width: isMobile ? "94vw" : undefined, maxWidth: "350px" },
    modalOverlay: { position: "fixed", inset: 0, backgroundColor: "rgba(0, 0, 0, 0.35)", display: "flex", justifyContent: "center", alignItems: "center", zIndex: 9999 },
    modal: { backgroundColor: "#f9f9f9", padding: isMobile ? 13 : 24, borderRadius: 12, maxWidth: isMobile ? "98vw" : 380, width: "100%", textAlign: "center", boxShadow: "0 0 24px #00000044" },
    modalTitle: { fontWeight: 700, fontSize: isMobile ? "1rem" : "1.2rem", marginBottom: isMobile ? 13 : 20, color: "#0d47a1" },
    modalInput: { width: "100%", padding: isMobile ? 7 : 10, marginBottom: 10, fontSize: 15, borderRadius: 6, border: "1.6px solid #0288d1", outline: "none" },
    modalBtn: { width: "100%", padding: isMobile ? 9 : 12, background: "linear-gradient(90deg,#0288d1,#0d47a1)", color: "white", fontWeight: 700, fontSize: isMobile ? "0.95rem" : "1rem", border: "none", borderRadius: 6, cursor: "pointer", marginTop: 7 },
    cancelBtn: { marginTop: 9, cursor: "pointer", color: "#0d47a1", fontWeight: 600, border: "none", background: "transparent", fontSize: isMobile ? "0.93rem" : undefined },
  };


  const [forgotData, setForgotData] = useState({
    email: '',
    dob: '',
    mobile: '',
    newPassword: '',
    confirmPassword: ''
  });

  const navigate = useNavigate();
  const [successMessage, setSuccessMessage] = useState(false);

  const handleChange = (e) =>
    setFormData({ ...formData, [e.target.name]: e.target.value });

  const onForgotChange = e =>
    setForgotData({ ...forgotData, [e.target.name]: e.target.value });

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('http://10.100.209.92:5000/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: formData.username, password: formData.password, role }),
      });
      const result = await response.json();
      if (result.success) {
        setSuccessMessage(true);
        setTimeout(() => {
          setSuccessMessage(false);
          navigate('/main_page');
        }, 2000);
      } else {
        alert(result.message || 'Login failed');
      }
    } catch (err) {
      alert('Server error');
    }
  };

  const handleForgotSubmit = async (e) => {
    e.preventDefault();
    if (forgotData.newPassword !== forgotData.confirmPassword) {
      alert("Passwords do not match");
      return;
    }
    try {
      const response = await fetch(' http://10.100.209.92:5000/api/forgot-password/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: forgotData.email,
          dob: forgotData.dob,
          mobile: forgotData.mobile,
          newPassword: forgotData.newPassword,
          confirmPassword: forgotData.confirmPassword,
          role,
        }),
      });
      const result = await response.json();
      if (result.success) {
        alert("Password changed successfully, please login.");
        setShowForgot(false);
        window.location.href = "/login";
      } else {
        alert(result.message);
      }
    } catch (error) {
      alert("Server error, please try again later.");
    }
  };

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <div style={styles.left}>
          <img src="/login_photo.jpg" style={styles.leftImg} alt="Login Template" />
        </div>
        <div style={styles.right}>
          <form onSubmit={handleLogin} autoComplete="off">
            <div style={styles.headerRow}>
              <img src="/login_photo.jpg" style={styles.logoImg} alt="Logo" />
              <div style={styles.loginTitle}>Login to VigilNet</div>
            </div>
            <div style={styles.toggleWrap}>
              <button type="button" style={styles.toggleBtn(role === 'admin')} onClick={() => setRole('admin')}>Admin</button>
              <button type="button" style={styles.toggleBtn(role === 'user')} onClick={() => setRole('user')}>User</button>
            </div>
            <div style={styles.inputWrap}>
              <FaUser size={17} color="#1e90ff" />
              <input
                style={styles.input}
                type="text"
                name="username"
                placeholder={role === 'admin' ? "Admin ID" : "Username"}
                value={formData.username}
                onChange={handleChange}
                autoComplete="username"
                required
              />
            </div>
            <div style={styles.inputWrap}>
              <FaLock size={17} color="#1e90ff" />
              <input
                style={styles.input}
                type="password"
                name="password"
                placeholder="Password"
                value={formData.password}
                onChange={handleChange}
                autoComplete="current-password"
                required
              />
            </div>
            <button style={styles.button} type="submit">
              LOGIN
            </button>
            <div style={{ width: "100%", textAlign: "center" }}>
              <span style={styles.forgot} onClick={() => setShowForgot(true)}>
                Forgot your Password?
              </span>
            </div>
            <div style={styles.welcome}>WELCOME</div>
            <div style={styles.signupWrap}>Don’t have an account?</div>
            <Link to="/signup" style={{ textDecoration: 'none' }}>
              <button style={styles.signup} type="button">Sign up</button>
            </Link>
          </form>

          {successMessage && (
            <div style={{
              position: 'fixed',
              top: 30,
              right: 30,
              background: 'linear-gradient(135deg, #00d4ff 0%, #0099ff 100%)',
              color: '#fff',
              padding: '18px 28px',
              borderRadius: '12px',
              fontSize: '1.1rem',
              fontWeight: '700',
              boxShadow: '0 8px 24px #0099ff55',
              zIndex: 9999,
              letterSpacing: '1px',
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              animation: 'slideInRight 0.5s ease-in-out',
            }}>
              ✓ Login Successfully! Redirecting...
            </div>
          )}

          {showForgot && (
            <div style={styles.modalOverlay}>
              <div style={styles.modal}>
                <div style={styles.modalTitle}>Forgot Password</div>
                <form onSubmit={handleForgotSubmit}>
                  <input
                    style={styles.modalInput}
                    type="email"
                    name="email"
                    placeholder="Email"
                    value={forgotData.email}
                    onChange={onForgotChange}
                    required
                  />
                  <input
                    style={styles.modalInput}
                    type="date"
                    name="dob"
                    placeholder="Date of Birth"
                    value={forgotData.dob}
                    onChange={onForgotChange}
                    required
                  />
                  <input
                    style={styles.modalInput}
                    type="text"
                    name="mobile"
                    placeholder="Mobile Number"
                    value={forgotData.mobile}
                    onChange={onForgotChange}
                    required
                  />
                  <input
                    style={styles.modalInput}
                    type="password"
                    name="newPassword"
                    placeholder="New Password"
                    value={forgotData.newPassword}
                    onChange={onForgotChange}
                    required
                  />
                  <input
                    style={styles.modalInput}
                    type="password"
                    name="confirmPassword"
                    placeholder="Confirm Password"
                    value={forgotData.confirmPassword}
                    onChange={onForgotChange}
                    required
                  />
                  <button type="submit" style={styles.modalBtn}>Change Password</button>
                </form>
                <button style={styles.cancelBtn} onClick={() => setShowForgot(false)}>Cancel</button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Login;


// import React, { useState } from "react";

// const styles = {
//   page: {
//     minHeight: "100vh",
//     width: "100vw",
//     background: "#131722",              // dark background
//     padding: "0",
//     fontFamily: "'Poppins', 'Segoe UI', Arial, sans-serif",
//     display: "flex",
//     flexDirection: "column",
//     alignItems: "center",
//   },
//   header: {
//     marginTop: "36px",
//     marginBottom: "24px",
//     fontWeight: "800",
//     fontSize: "2.35rem",
//     letterSpacing: "2.5px",
//     color: "#39a1f4",                  // blue main font
//     textShadow: "0 2px 14px #112359ad",
//   },
//   theory: {
//     background: "#18203a",
//     color: "#42bbf5",
//     padding: "28px",
//     borderRadius: "18px",
//     maxWidth: "820px",
//     lineHeight: 1.65,
//     fontSize: "1.08rem",
//     boxShadow: "0 2px 18px #11235917",
//     marginBottom: "32px"
//   },
//   filterBox: {
//     background: "#181d2a",
//     borderRadius: "18px",
//     boxShadow: "0 3px 15px #2a425a45",
//     padding: "24px 28px",
//     marginBottom: "20px",
//     display: "flex",
//     gap: "28px",
//     alignItems: "center",
//     flexWrap: "wrap",
//   },
//   select: {
//     padding: "10px 9px",
//     border: "1.7px solid #224577",
//     borderRadius: "8px",
//     background: "#20253b",
//     fontSize: "1.09rem",
//     fontWeight: 600,
//     color: "#58d4fe",                 // light blue font for select
//     minWidth: "110px"
//   },
//   label: {
//     color: "#9cc9ff",                 // pale blue
//     fontWeight: 600,
//     fontSize: "1.09rem",
//     marginRight: "6px"
//   },
//   graphBox: {
//     background: "#23273a",
//     marginTop: "16px",
//     width: "100%",
//     maxWidth: "900px",
//     minHeight: "340px",
//     borderRadius: "18px",
//     boxShadow: "0 2px 24px #1420631f",
//     padding: "36px 30px"
//   },
//   graphTitle: {
//     fontSize: "1.18rem",
//     color: "#72bbef",
//     fontWeight: 700,
//     marginBottom: "10px",
//     letterSpacing: ".7px"
//   }
// };

// const getDaysInMonth = (year, month) => {
//   return new Date(year, month, 0).getDate();
// };

// function Dashboard() {
//   // FROM/TO for date
//   const now = new Date();

//   // state for from
//   const [fromYear, setFromYear] = useState(now.getFullYear());
//   const [fromMonth, setFromMonth] = useState(now.getMonth() + 1);
//   const [fromDay, setFromDay] = useState(now.getDate());
//   const [fromHour, setFromHour] = useState(0);

//   // state for to
//   const [toYear, setToYear] = useState(now.getFullYear());
//   const [toMonth, setToMonth] = useState(now.getMonth() + 1);
//   const [toDay, setToDay] = useState(now.getDate());
//   const [toHour, setToHour] = useState(23);

//   // Replace below with your fetched/processed data!
//   const [data, setData] = useState([]);

//   // Generate options
//   const yearOptions = Array.from({ length: 6 }, (_, i) => 2020 + i);
//   const monthOptions = Array.from({ length: 12 }, (_, i) => i + 1);
//   const fromDayOptions = Array.from({ length: getDaysInMonth(fromYear, fromMonth) }, (_, i) => i + 1);
//   const toDayOptions = Array.from({ length: getDaysInMonth(toYear, toMonth) }, (_, i) => i + 1);
//   const hourOptions = Array.from({ length: 24 }, (_, i) => i);

//   const handleFilter = () => {
//     // Here, you would process crowd.txt with the filter values!
//     // setData(filteredData);
//     alert(
//       `Filtering data from ${fromDay}-${fromMonth}-${fromYear} ${fromHour}:00 to ${toDay}-${toMonth}-${toYear} ${toHour}:00`
//     );
//   };

//   return (
//     <div style={styles.page}>
//       <div style={styles.header}>VigilNet Historical Dashboard</div>
//       <div style={styles.theory}>
//         <strong>How does the Dashboard help?</strong><br />
//         The VigilNet dashboard provides interactive tools for tracking, analyzing, and predicting crowd events across time.<br />
//         Admins can select any date/time range to view crowd levels, spikes, and historical trends in the dataset. <br /><br />
//         <strong>How do the models work?</strong><br />
//         Deep learning and time series models (like LSTM and regression) study previous crowd densities, discovering patterns for both normal and surge events.<br />
//         These predictions help allocate resources, optimize security, and plan future events proactively.<br /><br />
//         <strong>Why visualize?</strong> <br />
//         Visual charts enable clear spotting of peaks, lulls, and incident correlations. With filters, users compare crowd profiles for days, months, hours—helping understand human behavior and predict future risks.
//       </div>
//       {/* Filter panel */}
//       <div style={styles.filterBox}>
//         <span style={styles.label}>From</span>
//         <select style={styles.select} value={fromDay} onChange={e => setFromDay(Number(e.target.value))}>
//           {fromDayOptions.map(d => <option key={d} value={d}>{d}</option>)}
//         </select>
//         <select style={styles.select} value={fromMonth} onChange={e => setFromMonth(Number(e.target.value))}>
//           {monthOptions.map(m => <option key={m} value={m}>{m}</option>)}
//         </select>
//         <select style={styles.select} value={fromYear} onChange={e => setFromYear(Number(e.target.value))}>
//           {yearOptions.map(y => <option key={y} value={y}>{y}</option>)}
//         </select>
//         <select style={styles.select} value={fromHour} onChange={e => setFromHour(Number(e.target.value))}>
//           {hourOptions.map(h => <option key={h} value={h}>{h}:00</option>)}
//         </select>
//         <span style={styles.label}>To</span>
//         <select style={styles.select} value={toDay} onChange={e => setToDay(Number(e.target.value))}>
//           {toDayOptions.map(d => <option key={d} value={d}>{d}</option>)}
//         </select>
//         <select style={styles.select} value={toMonth} onChange={e => setToMonth(Number(e.target.value))}>
//           {monthOptions.map(m => <option key={m} value={m}>{m}</option>)}
//         </select>
//         <select style={styles.select} value={toYear} onChange={e => setToYear(Number(e.target.value))}>
//           {yearOptions.map(y => <option key={y} value={y}>{y}</option>)}
//         </select>
//         <select style={styles.select} value={toHour} onChange={e => setToHour(Number(e.target.value))}>
//           {hourOptions.map(h => <option key={h} value={h}>{h}:00</option>)}
//         </select>
//         <button
//           style={{
//             ...styles.select,
//             background: "linear-gradient(90deg, #22309e 70%, #39a1f4 120%)",
//             color: "#fff",
//             cursor: "pointer",
//             border: "none",
//             fontWeight: 700,
//             padding: "11px 25px",
//             marginLeft: 10
//           }}
//           onClick={handleFilter}
//         >Filter Data</button>
//       </div>
//       {/* Graphical box */}
//       <div style={styles.graphBox}>
//         <div style={styles.graphTitle}>
//           Crowd Trends<br />
//           ({fromDay}-{fromMonth}-{fromYear} {fromHour}:00 <span style={{color:"#528fff", fontWeight:600}}>to</span> {toDay}-{toMonth}-{toYear} {toHour}:00)
//         </div>
//         <div style={{ color: "#3399ff", textAlign: "center", paddingTop: 60, fontSize: '1.03rem' }}>
//           {/* Replace this with your chart library/chart component */}
//           [Graphical representation – connect your chart/code here.]
//         </div>
//       </div>
//     </div>
//   );
// }

// export default Dashboard;

import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

// const styles = {
//   page: {
//     minHeight: "100vh",
//     width: "100vw",
//     background: "radial-gradient(circle at 20% 30%, #0c1022 0%, #0b0e1a 100%)",
//     fontFamily: "'Poppins', 'Segoe UI', sans-serif",
//     display: "flex",
//     flexDirection: "column",
//     alignItems: "center",
//     animation: "fadeIn 1s ease-in",
//     overflowX: "hidden",
//   },
//   header: {
//     marginTop: "50px",
//     marginBottom: "28px",
//     fontWeight: "800",
//     fontSize: "2.4rem",
//     letterSpacing: "2.3px",
//     color: "#4dc3ff",
//     textShadow: "0 2px 18px #0077ff90",
//     animation: "slideDown 1.1s ease-out",
//   },
//   backBtn: {
//     position: "absolute",
//     top: "25px",
//     left: "25px",
//     padding: "9px 18px",
//     background: "linear-gradient(135deg, #283e75 0%, #1a2b55 100%)",
//     color: "#9bd3ff",
//     border: "1px solid #395b94",
//     borderRadius: "8px",
//     fontSize: "0.95rem",
//     cursor: "pointer",
//     marginTop:"40px",
//     transition: "all 0.3s ease",
//   },
//   theory: {
//     background: "rgba(32, 44, 78, 0.45)",
//     color: "#a9d5ff",
//     padding: "30px",
//     borderRadius: "18px",
//     maxWidth: "900px",
//     lineHeight: 1.7,
//     fontSize: "1.05rem",
//     boxShadow: "0 4px 18px rgba(17, 37, 97, 0.25)",
//     backdropFilter: "blur(8px)",
//     marginBottom: "35px",
//     animation: "fadeUp 1.3s ease",
//   },
//   filterBox: {
//     background: "rgba(26, 36, 65, 0.7)",
//     borderRadius: "16px",
//     boxShadow: "0 4px 16px rgba(42, 66, 90, 0.35)",
//     padding: "22px 28px",
//     marginBottom: "24px",
//     display: "flex",
//     gap: "22px",
//     alignItems: "center",
//     flexWrap: "wrap",
//     animation: "fadeUp 1.4s ease-in-out",
//   },
//   select: {
//     padding: "9px 8px",
//     border: "1.5px solid #234b87",
//     borderRadius: "8px",
//     background: "#1b2240",
//     fontSize: "1rem",
//     fontWeight: 600,
//     color: "#78c7ff",
//     minWidth: "105px",
//     transition: "all 0.3s ease",
//   },
//   selectHover: {
//     borderColor: "#509cff",
//     boxShadow: "0 0 8px #3f9cff70",
//   },
//   label: {
//     color: "#b6d7ff",
//     fontWeight: 600,
//     fontSize: "1.05rem",
//     marginRight: "8px",
//   },
//   button: {
//     background: "linear-gradient(90deg, #1f57c9 0%, #38b6ff 100%)",
//     color: "#fff",
//     cursor: "pointer",
//     border: "none",
//     fontWeight: 700,
//     padding: "10px 25px",
//     marginLeft: 10,
//     borderRadius: "8px",
//     letterSpacing: "0.7px",
//     boxShadow: "0 3px 10px rgba(20, 98, 255, 0.3)",
//     transition: "all 0.3s ease",
//   },
//   buttonHover: {
//     boxShadow: "0 4px 18px rgba(0, 144, 255, 0.6)",
//     transform: "translateY(-2px)",
//   },
//   graphBox: {
//     background: "rgba(30, 38, 68, 0.7)",
//     width: "100%",
//     maxWidth: "920px",
//     minHeight: "340px",
//     borderRadius: "18px",
//     boxShadow: "0 4px 24px rgba(20, 38, 99, 0.25)",
//     padding: "36px 30px",
//     animation: "fadeUp 1.6s ease-in",
//   },
//   graphTitle: {
//     fontSize: "1.18rem",
//     color: "#8bc6ff",
//     fontWeight: 700,
//     marginBottom: "15px",
//     letterSpacing: "0.6px",
//   },
// };


function useWindowWidth() {
  const [width, setWidth] = useState(window.innerWidth);
  useEffect(() => {
    const handleResize = () => setWidth(window.innerWidth);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);
  return width;
}

function Dashboard() {
  const navigate = useNavigate();
  const now = new Date();

  const [fromYear, setFromYear] = useState(now.getFullYear());
  const [fromMonth, setFromMonth] = useState(now.getMonth() + 1);
  const [fromDay, setFromDay] = useState(now.getDate());
  const [fromHour, setFromHour] = useState(0);
  const [toYear, setToYear] = useState(now.getFullYear());
  const [toMonth, setToMonth] = useState(now.getMonth() + 1);
  const [toDay, setToDay] = useState(now.getDate());
  const [toHour, setToHour] = useState(23);
  const [hovered, setHovered] = useState(false);

  const yearOptions = Array.from({ length: 6 }, (_, i) => 2020 + i);
  const monthOptions = Array.from({ length: 12 }, (_, i) => i + 1);
  const daysInMonth = (year, month) => new Date(year, month, 0).getDate();
  const fromDayOptions = Array.from({ length: daysInMonth(fromYear, fromMonth) }, (_, i) => i + 1);
  const toDayOptions = Array.from({ length: daysInMonth(toYear, toMonth) }, (_, i) => i + 1);
  const hourOptions = Array.from({ length: 24 }, (_, i) => i);


  const width = useWindowWidth();
  const isMobile = width < 700;

  const styles = {
    page: {
      minHeight: "100vh",
      width: "100vw",
      background: "radial-gradient(circle at 20% 30%, #0c1022 0%, #0b0e1a 100%)",
      fontFamily: "'Poppins', 'Segoe UI', sans-serif",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      animation: "fadeIn 1s ease-in",
      overflowX: "hidden",
      padding: isMobile ? "0 2vw" : undefined,
    },
    header: {
      marginTop: isMobile ? "29px" : "50px",
      marginBottom: isMobile ? "19px" : "28px",
      fontWeight: "800",
      fontSize: isMobile ? "1.35rem" : "2.4rem",
      letterSpacing: "2.3px",
      color: "#4dc3ff",
      textShadow: "0 2px 18px #0077ff90",
      animation: "slideDown 1.1s ease-out",
      textAlign: "center"
    },
    backBtn: {
      position: "absolute",
      top: isMobile ? "9px" : "25px",
      left: isMobile ? "9px" : "25px",
      padding: isMobile ? "8px 11px" : "9px 18px",
      background: "linear-gradient(135deg, #283e75 0%, #1a2b55 100%)",
      color: "#9bd3ff",
      border: "1px solid #395b94",
      borderRadius: "8px",
      fontSize: "0.95rem",
      cursor: "pointer",
      marginTop: isMobile ? "18px" : "40px",
      transition: "all 0.3s ease",
      zIndex: 2,
    },
    theory: {
      background: "rgba(32, 44, 78, 0.45)",
      color: "#a9d5ff",
      padding: isMobile ? "16px" : "30px",
      borderRadius: "18px",
      maxWidth: isMobile ? "98vw" : "900px",
      lineHeight: 1.7,
      fontSize: isMobile ? "1rem" : "1.05rem",
      boxShadow: "0 4px 18px rgba(17, 37, 97, 0.25)",
      backdropFilter: "blur(8px)",
      marginBottom: isMobile ? "19px" : "35px",
      animation: "fadeUp 1.3s ease",
    },
    filterBox: {
      background: "rgba(26, 36, 65, 0.7)",
      borderRadius: "16px",
      boxShadow: "0 4px 16px rgba(42, 66, 90, 0.35)",
      padding: isMobile ? "13px 8px" : "22px 28px",
      marginBottom: isMobile ? "13px" : "24px",
      display: isMobile ? "block" : "flex",
      gap: isMobile ? "8px" : "22px",
      alignItems: "center",
      flexWrap: "wrap",
      animation: "fadeUp 1.4s ease-in-out",
    },
    select: {
      padding: isMobile ? "8px 4px" : "9px 8px",
      border: "1.5px solid #234b87",
      borderRadius: "8px",
      background: "#1b2240",
      fontSize: isMobile ? "0.93rem" : "1rem",
      fontWeight: 600,
      color: "#78c7ff",
      minWidth: "105px",
      transition: "all 0.3s ease",
      marginBottom: isMobile ? "9px" : "0",
    },
    selectHover: {
      borderColor: "#509cff",
      boxShadow: "0 0 8px #3f9cff70",
    },
    label: {
      color: "#b6d7ff",
      fontWeight: 600,
      fontSize: isMobile ? "0.98rem" : "1.05rem",
      marginRight: "8px",
    },
    button: {
      background: "linear-gradient(90deg, #1f57c9 0%, #38b6ff 100%)",
      color: "#fff",
      cursor: "pointer",
      border: "none",
      fontWeight: 700,
      padding: isMobile ? "8px 18px" : "10px 25px",
      marginLeft: isMobile ? 0 : 10,
      borderRadius: "8px",
      letterSpacing: "0.7px",
      boxShadow: "0 3px 10px rgba(20, 98, 255, 0.3)",
      transition: "all 0.3s ease",
      width: isMobile ? "99vw" : undefined,
      marginTop: isMobile ? "8px" : "0"
    },
    buttonHover: {
      boxShadow: "0 4px 18px rgba(0, 144, 255, 0.6)",
      transform: "translateY(-2px)",
    },
    graphBox: {
      background: "rgba(30, 38, 68, 0.7)",
      width: "100%",
      maxWidth: isMobile ? "99vw" : "920px",
      minHeight: isMobile ? "240px" : "340px",
      borderRadius: "18px",
      boxShadow: "0 4px 24px rgba(20, 38, 99, 0.25)",
      padding: isMobile ? "14px 7px" : "36px 30px",
      animation: "fadeUp 1.6s ease-in",
      marginBottom: isMobile ? "16px" : "0"
    },
    graphTitle: {
      fontSize: isMobile ? "1rem" : "1.18rem",
      color: "#8bc6ff",
      fontWeight: 700,
      marginBottom: isMobile ? "11px" : "15px",
      letterSpacing: "0.6px",
      textAlign: "center"
    },
  };


  const handleFilter = () => {
    alert(
      `Filtering data from ${fromDay}-${fromMonth}-${fromYear} ${fromHour}:00 to ${toDay}-${toMonth}-${toYear} ${toHour}:00`
    );
  };

  return (
    <div style={styles.page}>
      <button
        style={styles.backBtn}
        onMouseEnter={e => (e.target.style.background = "#273a6b")}
        onMouseLeave={e => (e.target.style.background = "linear-gradient(135deg, #283e75 0%, #1a2b55 100%)")}
        onClick={() => navigate(-1)}
      >
        ← Back
      </button>

      <div style={styles.header}>VigilNet Historical Dashboard</div>

      <div style={styles.theory}>
        <strong>How does the Dashboard help?</strong><br />
        The VigilNet dashboard provides interactive tools for tracking, analyzing, and predicting crowd events across time.<br />
        Admins can select any date/time range to view crowd levels, spikes, and historical trends in the dataset. <br /><br />
        <strong>How do the models work?</strong><br />
        Deep learning and time series models (like LSTM and regression) study previous crowd densities, discovering patterns for both normal and surge events.<br />
        These predictions help allocate resources, optimize security, and plan future events proactively.<br /><br />
        <strong>Why visualize?</strong> <br />
        Visual charts reveal peaks, lulls, and incident correlations. With filters, users compare crowd profiles by day, month, or time—empowering smarter predictions.
      </div>

      <div style={styles.filterBox}>
        <span style={styles.label}>From</span>
        <select style={styles.select} value={fromDay} onChange={e => setFromDay(Number(e.target.value))}>
          {fromDayOptions.map(d => <option key={d} value={d}>{d}</option>)}
        </select>
        <select style={styles.select} value={fromMonth} onChange={e => setFromMonth(Number(e.target.value))}>
          {monthOptions.map(m => <option key={m} value={m}>{m}</option>)}
        </select>
        <select style={styles.select} value={fromYear} onChange={e => setFromYear(Number(e.target.value))}>
          {yearOptions.map(y => <option key={y} value={y}>{y}</option>)}
        </select>
        <select style={styles.select} value={fromHour} onChange={e => setFromHour(Number(e.target.value))}>
          {hourOptions.map(h => <option key={h} value={h}>{h}:00</option>)}
        </select>

        <span style={styles.label}>To</span>
        <select style={styles.select} value={toDay} onChange={e => setToDay(Number(e.target.value))}>
          {toDayOptions.map(d => <option key={d} value={d}>{d}</option>)}
        </select>
        <select style={styles.select} value={toMonth} onChange={e => setToMonth(Number(e.target.value))}>
          {monthOptions.map(m => <option key={m} value={m}>{m}</option>)}
        </select>
        <select style={styles.select} value={toYear} onChange={e => setToYear(Number(e.target.value))}>
          {yearOptions.map(y => <option key={y} value={y}>{y}</option>)}
        </select>
        <select style={styles.select} value={toHour} onChange={e => setToHour(Number(e.target.value))}>
          {hourOptions.map(h => <option key={h} value={h}>{h}:00</option>)}
        </select>

        <button
          style={{
            ...styles.button,
            ...(hovered ? styles.buttonHover : {}),
          }}
          onMouseEnter={() => setHovered(true)}
          onMouseLeave={() => setHovered(false)}
          onClick={handleFilter}
        >
          Filter Data
        </button>
      </div>

      <div style={styles.graphBox}>
        <div style={styles.graphTitle}>
          Crowd Trends<br />
          ({fromDay}-{fromMonth}-{fromYear} {fromHour}:00 to {toDay}-{toMonth}-{toYear} {toHour}:00)
        </div>
        <div style={{ color: "#58b6ff", textAlign: "center", paddingTop: 60, fontSize: '1.03rem' }}>
          [Graphical representation – integrate chart component here.]
        </div>
      </div>

      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeUp {
          0% { opacity: 0; transform: translateY(30px); }
          100% { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideDown {
          from { transform: translateY(-40px); opacity: 0; }
          to { transform: translateY(0); opacity: 1; }
        }
      `}</style>
    </div>
  );
}

export default Dashboard;

import React, { useState, useEffect } from "react";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";
// const styles = {
//   page: {
//     background: "linear-gradient(135deg, #101a2b 0%, #002e53 100%)",
//     minHeight: "100vh",
//     padding: "60px 80px",
//     fontFamily: "'Poppins', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
//     color: "#eaf4fb",
//     display: "flex",
//     flexDirection: "column",
//     alignItems: "center",
//     gap: 40,
//     animation: "fadeInDown 0.8s ease forwards",
//   },
//   title: {
//     fontSize: 32,
//     fontWeight: "700",
//     color: "#59a3fc",
//     textShadow: "0 3px 15px #0353a4cc",
//     animation: "slideInLeft 0.8s ease forwards",
//   },
//   uploadCard: {
//     background: "linear-gradient(135deg, #161f38 60%, #183a63 100%)",
//     borderRadius: 20,
//     padding: 30,
//     boxShadow: "0 10px 28px rgba(1, 18, 40, 0.8)",
//     width: "500px",
//     textAlign: "center",
//     animation: "fadeInUp 1s ease forwards",
//   },
//   inputFile: {
//     marginTop: 24,
//     marginBottom: 24,
//     cursor: "pointer",
//     borderRadius: 12,
//     padding: "14px 20px",
//     border: "2px dashed #59a3fc",
//     backgroundColor: "#12264d",
//     color: "#93baff",
//     fontWeight: "600",
//     fontSize: 18,
//     width: "90%",
//     transition: "background-color 0.3s ease",
//   },
//   inputFileHover: {
//     backgroundColor: "#15336b",
//   },
//   previewImage: {
//     maxWidth: "100%",
//     maxHeight: 360,
//     borderRadius: 16,
//     boxShadow: "0 4px 18px #12386aaa",
//   },
//   resultBox: {
//     marginTop: 30,
//     fontSize: 22,
//     fontWeight: "700",
//     color: "#80b1ff",
//     textShadow: "0 2px 14px #0c3a85bb",
//     animation: "fadeIn 1s ease forwards",
//   },
//   submitBtn: {
//     marginTop: 16,
//     background: "linear-gradient(135deg, #025dfe 0%, #161f38 100%)",
//     color: "#fff",
//     borderRadius: 24,
//     padding: "14px 36px",
//     fontSize: 20,
//     fontWeight: "700",
//     cursor: "pointer",
//     border: "none",
//     boxShadow: "0 10px 32px #125bc699",
//     transition: "transform 0.25s ease, box-shadow 0.25s ease",
//   },
//   submitBtnDisabled: {
//     background: "#26456a",
//     cursor: "not-allowed",
//     boxShadow: "none",
//   },
//   resultBox: {
//     marginTop: 8,
//     padding: 10,
//     borderRadius: 6,
//     backgroundColor: "#15336b",
//     color: "#80b1ff",
//     width: "100%",
//     textAlign: "center",
//   },
//   '@keyframes fadeInDown': {
//     from: { opacity: 0, transform: 'translateY(-30px)' },
//     to: { opacity: 1, transform: 'translateY(0)' },
//   },
//   '@keyframes fadeInUp': {
//     from: { opacity: 0, transform: 'translateY(30px)' },
//     to: { opacity: 1, transform: 'translateY(0)' },
//   },
//   '@keyframes fadeIn': {
//     from: { opacity: 0 },
//     to: { opacity: 1 },
//   },
//   '@keyframes slideInLeft': {
//     from: { opacity: 0, transform: 'translateX(-40px)' },
//     to: { opacity: 1, transform: 'translateX(0)' },
//   },
// };

// function CrowdCountPhoto() {
//   const [selectedFile, setSelectedFile] = useState(null);
//   const [previewUrl, setPreviewUrl] = useState(null);
//   const [countResult, setCountResult] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const [hoverFileInput, setHoverFileInput] = useState(false);

//   const API_ENDPOINT = "http://127.0.0.1:8000/crowdcount";// Replace with your backend URL

//   const handleFileChange = (event) => {
//     if (event.target.files && event.target.files[0]) {
//       const file = event.target.files[0];
//       setSelectedFile(file);
//       setCountResult(null);
//       setPreviewUrl(URL.createObjectURL(file));
//     }
//   };

//   const handleSubmit = async () => {
//     if (!selectedFile) return;

//     setLoading(true);
//     setCountResult(null);

//     try {
//       const formData = new FormData();
//       formData.append("image", selectedFile);

//       const response = await fetch(API_ENDPOINT, {
//         method: "POST",
//         body: formData,
//       });

//       if (!response.ok) {
//         throw new Error("Server error");
//       }

//       const data = await response.json();
//       // Assuming response contains { count: number }
//       setCountResult(data.count);
//     } catch (error) {
//       setCountResult("Error: Unable to count people");
//       console.error(error);
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div style={styles.page}>
//       <h1 style={styles.title}>Crowd Counting from Photo</h1>

//       <div style={styles.uploadCard}>
//         <input
//           type="file"
//           accept="image/*"
//           style={{
//             ...styles.inputFile,
//             ...(hoverFileInput ? styles.inputFileHover : {}),
//           }}
//           onMouseEnter={() => setHoverFileInput(true)}
//           onMouseLeave={() => setHoverFileInput(false)}
//           onChange={handleFileChange}
//         />

//         {previewUrl && (
//           <img
//             src={previewUrl}
//             alt="Uploaded preview"
//             style={styles.previewImage}
//           />
//         )}

//         <button
//           style={{
//             ...styles.submitBtn,
//             ...(loading || !selectedFile ? styles.submitBtnDisabled : {}),
//           }}
//           onClick={handleSubmit}
//           disabled={loading || !selectedFile}
//         >
//           {loading ? "Counting..." : "Count People"}
//         </button>

//         {countResult !== null && !loading && (
//           <div style={styles.resultBox}>
//             {typeof countResult === "number"
//               ? `Estimated number of people: ${countResult}`
//               : countResult}
//           </div>
//         )}
//       </div>
//     </div>
//   );
// }

// export default CrowdCountPhoto;

function useWindowWidth() {
  const [width, setWidth] = useState(window.innerWidth);
  useEffect(() => {
    const handleResize = () => setWidth(window.innerWidth);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);
  return width;
}


function CrowdCountPhoto() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [countResult, setCountResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [hoverFileInput, setHoverFileInput] = useState(false);
  const [connStatus, setConnStatus] = useState(null);
  const [connMessage, setConnMessage] = useState("");

  const width = useWindowWidth();
const isMobile = width < 700;

const styles = {
  page: {
    background: "linear-gradient(135deg, #101a2b 0%, #002e53 100%)",
    minHeight: "100vh",
    padding: isMobile ? "28px 4vw" : "60px 80px",
    fontFamily: "'Poppins', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    color: "#eaf4fb",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: isMobile ? 18 : 40,
    animation: "fadeInDown 0.8s ease forwards",
    width: "100vw"
  },
  title: {
    fontSize: isMobile ? "1.5rem" : 32,
    fontWeight: "700",
    color: "#59a3fc",
    textShadow: "0 3px 15px #0353a4cc",
    animation: "slideInLeft 0.8s ease forwards",
    textAlign: "center",
  },
  uploadCard: {
    background: "linear-gradient(135deg, #161f38 60%, #183a63 100%)",
    borderRadius: 20,
    padding: isMobile ? 18 : 30,
    boxShadow: "0 10px 28px rgba(1, 18, 40, 0.8)",
    width: isMobile ? "98vw" : "500px",
    textAlign: "center",
    animation: "fadeInUp 1s ease forwards",
    maxWidth: "500px",
    margin: "0 auto"
  },
  inputFile: {
    marginTop: isMobile ? 14 : 24,
    marginBottom: isMobile ? 14 : 24,
    cursor: "pointer",
    borderRadius: 12,
    padding: isMobile ? "10px 10px" : "14px 20px",
    border: "2px dashed #59a3fc",
    backgroundColor: "#12264d",
    color: "#93baff",
    fontWeight: "600",
    fontSize: isMobile ? 15 : 18,
    width: isMobile ? "90vw" : "90%",
    transition: "background-color 0.3s ease",
    maxWidth: "460px"
  },
  inputFileHover: {
    backgroundColor: "#15336b",
  },
  previewImage: {
    maxWidth: "100%",
    maxHeight: isMobile ? 190 : 360,
    borderRadius: 16,
    boxShadow: "0 4px 18px #12386aaa",
    marginBottom: isMobile ? 11 : 0,
  },
  resultBox: {
    marginTop: isMobile ? 8 : 30,
    padding: isMobile ? 7 : 10,
    borderRadius: 6,
    backgroundColor: "#15336b",
    color: "#80b1ff",
    width: "100%",
    textAlign: "center",
    fontSize: isMobile ? "1rem" : 22,
  },
  submitBtn: {
    marginTop: 16,
    background: "linear-gradient(135deg, #025dfe 0%, #161f38 100%)",
    color: "#fff",
    borderRadius: 24,
    padding: isMobile ? "11px 14px" : "14px 36px",
    fontSize: isMobile ? "1rem" : 20,
    fontWeight: "700",
    cursor: "pointer",
    border: "none",
    boxShadow: "0 10px 32px #125bc699",
    transition: "transform 0.25s ease, box-shadow 0.25s ease",
    width: isMobile ? "90vw" : undefined,
    maxWidth: "380px"
  },
  submitBtnDisabled: {
    background: "#26456a",
    cursor: "not-allowed",
    boxShadow: "none",
  },
  '@keyframes fadeInDown': {
    from: { opacity: 0, transform: 'translateY(-30px)' },
    to: { opacity: 1, transform: 'translateY(0)' },
  },
  '@keyframes fadeInUp': {
    from: { opacity: 0, transform: 'translateY(30px)' },
    to: { opacity: 1, transform: 'translateY(0)' },
  },
  '@keyframes fadeIn': {
    from: { opacity: 0 },
    to: { opacity: 1 },
  },
  '@keyframes slideInLeft': {
    from: { opacity: 0, transform: 'translateX(-40px)' },
    to: { opacity: 1, transform: 'translateX(0)' },
  },
};


  const API_BASE = "http://127.0.0.1:8000";
  const API_ENDPOINT = `${API_BASE}/crowdcount`;
  const OPENAPI_URL = `${API_BASE}/openapi.json`;

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleFileChange = (event) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setSelectedFile(file);
      setCountResult(null);
      setPreviewUrl(URL.createObjectURL(file));
      setConnStatus(null);
      setConnMessage("");
    }
  };

  const checkConnection = async (timeoutMs = 5000) => {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const resp = await fetch(OPENAPI_URL, { method: "GET", signal: controller.signal });
      clearTimeout(id);
      if (!resp.ok) {
        const text = await resp.text().catch(() => "");
        throw new Error(`HTTP ${resp.status} ${text}`);
      }
      await resp.json();
      setConnStatus("ok");
      setConnMessage("Backend reachable");
      return true;
    } catch (err) {
      clearTimeout(id);
      setConnStatus("error");
      if (err.name === "AbortError") {
        setConnMessage("Connection timed out (check backend or port).");
      } else {
        setConnMessage(`Connection failed: ${err.message}`);
      }
      return false;
    }
  };

  const handleSubmit = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setCountResult(null);
    setConnStatus(null);
    setConnMessage("");
    const ok = await checkConnection(5000);
    if (!ok) {
      setLoading(false);
      return;
    }
    try {
      const formData = new FormData();
      formData.append("image", selectedFile);
      const response = await fetch(API_ENDPOINT, { method: "POST", body: formData });
      if (!response.ok) {
        let errMsg = `Server error: ${response.status}`;
        try {
          const text = await response.text();
          const j = JSON.parse(text);
          errMsg = j.detail || text || errMsg;
        } catch (e) { }
        throw new Error(errMsg);
      }
      const data = await response.json();
      setCountResult({
        count_int: data.count,
        count_float:
          typeof data.count_float === "number"
            ? data.count_float
            : Number(data.count_float),
      });
    } catch (error) {
      setCountResult({
        error: true,
        message: error.message || "Error: Unable to count people",
      });
    } finally {
      setLoading(false);
    }
  };

  // --- PDF Download Handler ---
  const handleDownloadPdf = async () => {
    const doc = new jsPDF({
      orientation: "portrait",
      unit: "pt",
      format: "a4"
    });
    let y = 40;
    let x = 40;
    // Logo
    const logoImg = new window.Image();
    logoImg.src = "/logo.png";
    await new Promise((res) => { logoImg.onload = res; logoImg.onerror = res; });
    if (logoImg.width)
      doc.addImage(logoImg, "PNG", x, y, 60, 60);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(44);
    doc.setTextColor("#0A2342");
    doc.text("VigilNet", x + 80, y + 38, { align: "left" });
    doc.setFont("helvetica", "normal");
    doc.setFontSize(20);
    doc.setTextColor("#1E90FF");
    doc.text("Intelligent Crowd Monitoring & Instant Alerts", x + 80, y + 65, { align: "left" });
    doc.setDrawColor("#007BFF");
    doc.setLineWidth(2);
    doc.line(x + 80, y + 75, x + 380, y + 75);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(13);
    doc.setTextColor("#333333");
    doc.text(
      "Detect crowd density, abnormal motion, and camera tampering in real time.\nSend alerts to staff, automate workflows, and keep venues safe with low-latency models.",
      x, 140, { align: "left", maxWidth: 520 }
    );
    let imgY = 200;
    if (previewUrl) {
      const img = new window.Image();
      img.src = previewUrl;
      await new Promise((res) => { img.onload = res; img.onerror = res; });
      doc.addImage(img, "JPEG", x, imgY, 220, 170);
      doc.setFont("helvetica", "italic");
      doc.setFontSize(12);
      doc.setTextColor("#0A2342");
      doc.text("This is the uploaded image", x, imgY + 185, { align: "left" });
    }
    doc.setFont("helvetica", "bold");
    doc.setFontSize(22);
    doc.setTextColor("#0A2342");
    let resultY = imgY + 220;
    // if (countResult && !countResult.error) {
    //   doc.text(
    //     `Estimated (rounded): ${countResult.count_int ? countResult.count_int.toLocaleString() : ""}`,
    //     x,
    //     resultY,
    //     { align: "left" }
    //   );
    //   if (
    //     countResult.count_float !== undefined &&
    //     !isNaN(countResult.count_float)
    //   ) {
    //     doc.text(
    //       `Estimated (raw): ${Number(countResult.count_float).toFixed(2)}`,
    //       x,
    //       resultY + 30,
    //       { align: "left" }
    //     );
    //   }
    // }
    if (countResult && !countResult.error) {
      const roundedDiv10 = countResult.count_int
        ? Math.round(countResult.count_int / 10)
        : null;
      const rawDiv10 = countResult.count_float !== undefined && !isNaN(countResult.count_float)
        ? (Number(countResult.count_float) / 10).toFixed(2)
        : null;

      doc.text(
        `Estimated (rounded): ${roundedDiv10 !== null ? roundedDiv10.toLocaleString() : "Not available"}`,
        x,
        resultY,
        { align: "left" }
      );
      doc.text(
        `Estimated (raw): ${rawDiv10 !== null ? rawDiv10 : "Not available"}`,
        x,
        resultY + 30,
        { align: "left" }
      );
    }

    // Add digital sign image bottom-right
    const signImg = new window.Image();
    signImg.src = "/digital_sign.png";
    await new Promise((res) => { signImg.onload = res; signImg.onerror = res; });

    if (signImg.width) {
      const signWidth = 120;
      const signHeight = (signImg.height / signImg.width) * signWidth;

      doc.addImage(
        signImg,
        "PNG", // Use "PNG" for .png file
        doc.internal.pageSize.getWidth() - signWidth - 60,
        doc.internal.pageSize.getHeight() - signHeight - 60,
        signWidth,
        signHeight
      );
    }

    const pageHeight = doc.internal.pageSize.getHeight();
    doc.setDrawColor("#007BFF");
    doc.setLineWidth(2);
    doc.line(50, pageHeight - 40, doc.internal.pageSize.getWidth() - 50, pageHeight - 40);
    doc.setFont("times", "italic");
    doc.setFontSize(14);
    doc.setTextColor("#1E90FF");
    doc.text(
      "VigilNet Verified Digital Sign",
      doc.internal.pageSize.getWidth() / 2,
      pageHeight - 20,
      { align: "center" }
    );
    doc.save("VigilNet_Crowd_Report.pdf");
  };

  const downloadPdfBtnStyle = {
    padding: "12px 32px",
    backgroundColor: "#1E90FF",
    color: "#fff",
    fontWeight: 600,
    border: "none",
    borderRadius: "7px",
    fontSize: "18px",
    cursor: "pointer",
    margin: "15px 0 0 0",
    boxShadow: "0 2px 6px rgba(10, 35, 66, 0.10)",
    transition: "background 0.2s",
  };



  return (
    <div style={styles.page}>
      <h1 style={styles.title}>Crowd Counting from Photo</h1>
      <div style={styles.uploadCard}>
        <input
          type="file"
          accept="image/*"
          style={{
            ...styles.inputFile,
            ...(hoverFileInput ? styles.inputFileHover : {}),
          }}
          onMouseEnter={() => setHoverFileInput(true)}
          onMouseLeave={() => setHoverFileInput(false)}
          onChange={handleFileChange}
        />
        {previewUrl && (
          <img src={previewUrl} alt="Uploaded preview" style={styles.previewImage} />
        )}
        <button
          style={{
            ...styles.submitBtn,
            ...(loading || !selectedFile ? styles.submitBtnDisabled : {}),
          }}
          onClick={handleSubmit}
          disabled={loading || !selectedFile}
        >
          {loading ? "Counting..." : "Count People"}
        </button>
        {countResult !== null && !loading && (
          <div style={styles.resultBox}>
            {countResult.error ? (
              <span>{countResult.message}</span>
            ) : (
              // <div>
              //   <div>
              //     <strong>Estimated (rounded):</strong>{" "}
              //     {countResult.count_int
              //       ? countResult.count_int.toLocaleString()
              //       : "Not available"}
              //   </div>
              //   <div>
              //     <strong>Estimated (raw):</strong>{" "}
              //     {countResult.count_float !== undefined && !isNaN(countResult.count_float)
              //       ? Number(countResult.count_float).toFixed(2)
              //       : "Not available"}
              //   </div>
              // </div>
              <div>
                <div>
                  <strong>Estimated (rounded):</strong>{" "}
                  {countResult.count_int
                    ? Math.round(countResult.count_int / 10).toLocaleString()
                    : "Not available"}
                </div>
                <div>
                  <strong>Estimated (raw):</strong>{" "}
                  {countResult.count_float !== undefined && !isNaN(countResult.count_float)
                    ? (Number(countResult.count_float) / 10).toFixed(2)
                    : "Not available"}
                </div>
              </div>
            )}
          </div>
        )}
        {countResult && !loading && !countResult.error && (
          <button style={downloadPdfBtnStyle} onClick={handleDownloadPdf}>
            Download PDF Report
          </button>
        )}
        {connStatus !== null && (
          <div style={{ marginTop: 8, color: connStatus === "ok" ? "green" : "crimson" }}>
            {connMessage}
          </div>
        )}
      </div>
    </div>
  );
}

export default CrowdCountPhoto;

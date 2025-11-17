import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";

// const styles = {
//   /* Main background and structure */
//   page: {
//     background: "linear-gradient(135deg, #101a2b 0%, #002e53 100%)",
//     minHeight: "100vh",
//     padding: "48px 72px",
//     fontFamily: "'Poppins', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
//     color: "#eaf4fb",
//     display: "flex",
//     flexDirection: "column",
//     gap: 36,
//     position: "relative",
//     animation: "fadeInDown 0.7s",
//   },

//   /* Header section */
//   welcomeRow: {
//     display: "flex",
//     justifyContent: "space-between",
//     alignItems: "center",
//     marginBottom: 32,
//     animation: "fadeIn 1s",
//   },
//   welcomeText: {
//     fontSize: 28,
//     fontWeight: "700",
//     color: "#59a3fc",
//     letterSpacing: "0.03em",
//     animation: "slideInLeft 0.7s",
//     textShadow: "0 2px 14px #00396d88",
//   },

//   /* Account */
//   accountType: {
//     position: "relative",
//     cursor: "pointer",
//     padding: "10px 22px",
//     backgroundColor: "#162b44",
//     color: "#59a3fc",
//     borderRadius: 16,
//     fontWeight: "600",
//     boxShadow: "0 4px 22px #21518a22",
//     userSelect: "none",
//     border: "1.5px solid #21518a",
//     transition: "background-color 0.3s, box-shadow 0.3s, color 0.3s",
//   },
//   accountTypeHover: {
//     backgroundColor: "#2955a5",
//     color: "#fff",
//     boxShadow: "0 10px 32px #125bc699",
//   },
//   accountDetailsDropdown: {
//     position: "absolute",
//     top: "110%",
//     right: 0,
//     background: "#181f3a",
//     boxShadow: "0 8px 32px #01152fbb",
//     borderRadius: 16,
//     padding: 20,
//     width: 320,
//     zIndex: 20,
//     animation: "fadeInUp 0.5s",
//     color: "#eaf4fb",
//     fontWeight: "500",
//     border: "1px solid #264686",
//   },

//   label: {
//     fontWeight: "700",
//     color: "#59a3fc",
//     marginBottom: 8,
//     fontSize: 15,
//     display: "block",
//   },

//   /* Camera Section */
//   cameraPairsContainer: {
//     display: "flex",
//     flexDirection: "column",
//     gap: 32,
//     flexWrap: "nowrap",
//     animation: "fadeIn 1s",
//   },
//   cameraPair: {
//     display: "flex",
//     gap: 28,
//     alignItems: "flex-start",
//   },
//   cameraCard: {
//     background: "linear-gradient(135deg, #161f38 60%, #183a63 100%)",
//     borderRadius: 20,
//     boxShadow: "0 10px 38px #081d2e55",
//     overflow: "hidden",
//     paddingBottom: 20,
//     animation: "fadeInUp 1s",
//     flex: "1 1 49%",
//     minWidth: 360,
//     position: "relative",
//     border: "2px solid #21518a",
//     transition: "transform 0.3s, box-shadow 0.3s",
//   },
//   cameraCardHover: {
//     transform: "scale(1.07)",
//     boxShadow: "0 15px 40px #1a81fa55",
//   },
//   cameraTitle: {
//     fontSize: 22,
//     fontWeight: "700",
//     padding: "20px 24px 12px",
//     color: "#59a3fc",
//     borderBottom: "2px solid #284873",
//     background: "rgba(24,34,64, 0.78)",
//     letterSpacing: "0.04em",
//   },
//   cameraFrame: {
//     width: "100%",
//     height: 410,
//     backgroundColor: "#222c43",
//     borderRadius: 16,
//     display: "flex",
//     justifyContent: "center",
//     alignItems: "center",
//     overflow: "hidden",
//     position: "relative",
//     boxShadow: "0 2px 12px #162b44b1",
//   },
//   cameraVideo: {
//     width: "100%",
//     height: "100%",
//     objectFit: "contain",
//     borderRadius: "14px",
//     backgroundColor: "#000",
//   },
//   cameraSettings: {
//     marginTop: 18,
//     padding: "0 28px",
//   },
//   labelInline: {
//     fontWeight: "700",
//     marginRight: 14,
//     fontSize: 17,
//     color: "#59a3fc",
//     display: "inline-block",
//   },
//   volumeDisplay: {
//     fontWeight: "700",
//     marginLeft: 10,
//     color: "#93baff",
//     fontSize: 15,
//   },
//   /* Camera Controls */
//   toggleSwitch: {
//     marginTop: 18,
//     display: "flex",
//     alignItems: "center",
//     justifyContent: "space-between",
//     padding: "0 26px",
//   },
//   toggleLabel: {
//     fontWeight: "700",
//     fontSize: 17,
//     color: "#59a3fc",
//   },

//   /* Add Camera Button */
//   addCameraBtn: {
//     marginTop: 24,
//     background: "linear-gradient(135deg, #025dfe 0%, #161f38 100%)",
//     color: "#fff",
//     borderRadius: 24,
//     padding: "16px 48px",
//     fontSize: 20,
//     fontWeight: "800",
//     cursor: "pointer",
//     border: "none",
//     boxShadow: "0 8px 27px #025dfe44",
//     transition: "transform 0.3s, box-shadow 0.3s, background 0.3s",
//   },
//   addCameraBtnHover: {
//     transform: "translateY(-6px)",
//     boxShadow: "0 16px 48px #025dfe88",
//     background: "linear-gradient(135deg, #119aff 0%, #24375c 100%)",
//   },

//   /* Delete Camera Button */
//   deleteCameraBtn: {
//     position: "absolute",
//     top: 25,
//     right: 18,
//     background: "linear-gradient(135deg, #eb3349 0%, #89216b 100%)",
//     border: "none",
//     borderRadius: 8,
//     color: "#fff",
//     cursor: "pointer",
//     padding: "7px 15px",
//     fontWeight: "700",
//     fontSize: 13,
//     zIndex: 10,
//     boxShadow: "0 4px 12px #eb334988",
//     transition: "background 0.3s",
//   },
//   deleteCameraBtnHover: {
//     background: "linear-gradient(135deg, #861657 0%, #eb3349 100%)",
//   },

//   /* Action Buttons Row */
//   buttonRow: {
//     marginTop: 54,
//     display: "flex",
//     justifyContent: "center",
//     gap: 36,
//     flexWrap: "wrap",
//     animation: "fadeInUp 1s",
//   },
//   actionButton: {
//     background: "linear-gradient(135deg, #2962ff 0%, #161f38 100%)",
//     color: "#fff",
//     borderRadius: 22,
//     padding: "17px 46px",
//     fontSize: 21,
//     fontWeight: "900",
//     cursor: "pointer",
//     border: "none",
//     boxShadow: "0 9px 32px #2962ff44",
//     transition: "transform 0.3s, box-shadow 0.3s, background 0.3s",
//     letterSpacing: '0.03em',
//   },
//   actionButtonHover: {
//     transform: "translateY(-6px)",
//     boxShadow: "0 18px 48px #1969fa99",
//     background: "linear-gradient(135deg, #4e7cfa 0%, #101a2b 100%)",
//   },

//   /* Alert Panel */
//   alertPanel: {
//     position: "fixed",
//     top: 132,
//     right: 28,
//     width: 370,
//     background: "linear-gradient(135deg, #1d2337 80%, #1756cb 100%)",
//     boxShadow: "0 10px 44px #01152fbb",
//     borderRadius: 26,
//     padding: 26,
//     zIndex: 90,
//     animation: "fadeInUp 0.8s",
//     color: "#eaf4fb",
//     fontFamily: "'Poppins', 'Segoe UI'",
//     border: "1.5px solid #274dc8",
//   },
//   alertTitle: {
//     fontWeight: "900",
//     fontSize: 24,
//     marginBottom: 18,
//     color: "#59a3fc",
//     letterSpacing: '0.02em'
//   },
//   soundOption: {
//     marginBottom: 14,
//     fontSize: 17,
//     fontWeight: "600",
//     color: "#93baff",
//     display: 'inline-block'
//   },
//   volumeInput: {
//     width: "100%",
//     height: 7,
//     borderRadius: 6,
//     background: "#002e53",
//     cursor: "pointer",
//     boxShadow: "0 2px 10px #59a3fc44"
//   },
//   cameraNameInput: {
//     width: "100%",
//     padding: 11,
//     fontSize: 18,
//     borderRadius: 12,
//     border: "2px solid #284873",
//     marginBottom: 12,
//     background: "#122046",
//     color: "#93baff",
//     fontWeight: "700",
//     transition: "border-color 0.3s, box-shadow 0.3s",
//     boxShadow: "0 2px 14px #063e7277"
//   },
//   cameraNameInputFocus: {
//     borderColor: "#59a3fc",
//     background: "#13264d",
//     boxShadow: "0 0 14px #3467b2dd",
//     outline: "none",
//   },

//   /* Keyframes (for use in emotion/CSS-in-JS or inject in a <style> tag) */
//   "@keyframes fadeInDown": {
//     from: { opacity: 0, transform: "translateY(-30px)" },
//     to: { opacity: 1, transform: "translateY(0)" },
//   },
//   "@keyframes fadeInUp": {
//     from: { opacity: 0, transform: "translateY(40px)" },
//     to: { opacity: 1, transform: "translateY(0)" },
//   },
//   "@keyframes fadeIn": {
//     from: { opacity: 0 },
//     to: { opacity: 1 },
//   },
//   "@keyframes slideInLeft": {
//     from: { opacity: 0, transform: "translateX(-60px)" },
//     to: { opacity: 1, transform: "translateX(0)" },
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

function MainPage({ user }) {
  const navigate = useNavigate();

  const [accountOpen, setAccountOpen] = useState(false);
  const [alertOpen, setAlertOpen] = useState(false);
  const [systemSettingsOpen, setSystemSettingsOpen] = useState(false);


  const accountRef = useRef(null);
  const alertRef = useRef(null);
  const systemSettingsRef = useRef(null);


  const [selectedSound, setSelectedSound] = useState("sound1");
  const [volume, setVolume] = useState(50);


  const [selectedSystemCamera, setSelectedSystemCamera] = useState("");
  const [uploadedVideoURL, setUploadedVideoURL] = useState(null);


  const width = useWindowWidth();
  const isMobile = width < 700;

  const styles = {
    page: {
      background: "linear-gradient(135deg, #101a2b 0%, #002e53 100%)",
      minHeight: "100vh",
      padding: isMobile ? "18px 4vw" : "48px 72px",
      fontFamily: "'Poppins', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
      color: "#eaf4fb",
      display: "flex",
      flexDirection: "column",
      gap: isMobile ? 18 : 36,
      position: "relative",
      animation: "fadeInDown 0.7s",
      width: "90vw"
    },
   welcomeRow: {
      display: isMobile ? "block" : "flex",
      justifyContent: "space-between",
      alignItems: isMobile ? "flex-start" : "center",
      marginBottom: isMobile ? 17 : 32,
      animation: "fadeIn 1s",
      width: "100%"
    },
    welcomeText: {
      fontSize: isMobile ? 17 : 28,
      fontWeight: "700",
      color: "#59a3fc",
      letterSpacing: "0.03em",
      animation: "slideInLeft 0.7s",
      textShadow: "0 2px 14px #00396d88",
      marginBottom: isMobile ? 8 : 0,
      textAlign: isMobile ? "center" : "left"
    },

    accountType: {
      position: "relative",
      cursor: "pointer",
      padding: isMobile ? "10px 13vw" : "10px 22px",
      backgroundColor: "#162b44",
      color: "#59a3fc",
      borderRadius: 16,
      fontWeight: "600",
      boxShadow: "0 4px 22px #21518a22",
      userSelect: "none",
      border: "1.5px solid #21518a",
      transition: "background-color 0.3s, box-shadow 0.3s, color 0.3s",
      fontSize: isMobile ? "1.01rem" : undefined,
      display: isMobile ? "block" : undefined,
      marginBottom: isMobile ? 13 : 0
    },
    accountTypeHover: {
      backgroundColor: "#2955a5",
      color: "#fff",
      boxShadow: "0 10px 32px #125bc699",
    },
    accountDetailsDropdown: {
      position: "absolute",
      top: "110%",
      right: 0,
      background: "#181f3a",
      boxShadow: "0 8px 32px #01152fbb",
      borderRadius: 16,
      padding: isMobile ? 13 : 20,
      width: isMobile ? "94vw" : 320,
      zIndex: 20,
      animation: "fadeInUp 0.5s",
      color: "#eaf4fb",
      fontWeight: "500",
      border: "1px solid #264686",
    },

    label: {
      fontWeight: "700",
      color: "#59a3fc",
      marginBottom: 8,
      fontSize: isMobile ? 13 : 15,
      display: "block",
    },

    cameraPairsContainer: {
      display: "flex",
      flexDirection: "column",
      gap: isMobile ? 12 : 32,
      flexWrap: "nowrap",
      animation: "fadeIn 1s",
      width: "100%",
    },
    cameraPair: {
      display: isMobile ? "block" : "flex",
      gap: isMobile ? 11 : 28,
      alignItems: isMobile ? "stretch" : "flex-start",
      width: "100%",
      marginBottom: isMobile ? 17 : 0
    },
    cameraCard: {
      background: "linear-gradient(135deg, #161f38 60%, #183a63 100%)",
      borderRadius: 20,
      boxShadow: "0 10px 38px #081d2e55",
      overflow: "hidden",
      paddingBottom: 20,
      animation: "fadeInUp 1s",
      flex: isMobile ? "unset" : "1 1 49%",
      minWidth: isMobile ? "95vw" : 360,
      maxWidth: isMobile ? "96vw" : undefined,
      position: "relative",
      border: "2px solid #21518a",
      transition: "transform 0.3s, box-shadow 0.3s",
      margin: isMobile ? "0 0 13px 0" : undefined
    },
    cameraCardHover: {
      transform: isMobile ? "none" : "scale(1.07)",
      boxShadow: "0 15px 40px #1a81fa55",
    },
    cameraTitle: {
      fontSize: isMobile ? 15 : 22,
      fontWeight: "700",
      padding: isMobile ? "11px 10px 8px" : "20px 24px 12px",
      color: "#59a3fc",
      borderBottom: "2px solid #284873",
      background: "rgba(24,34,64, 0.78)",
      letterSpacing: "0.04em",
      textAlign: isMobile ? "center" : "left"
    },
    cameraFrame: {
      width: "100%",
      height: isMobile ? 180 : 410,
      backgroundColor: "#222c43",
      borderRadius: 16,
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      overflow: "hidden",
      position: "relative",
      boxShadow: "0 2px 12px #162b44b1",
      marginBottom: isMobile ? 7 : 0,
    },
    cameraVideo: {
      width: "100%",
      height: "100%",
      objectFit: "contain",
      borderRadius: "14px",
      backgroundColor: "#000",
    },
    cameraSettings: {
      marginTop: 18,
      padding: isMobile ? "0 3vw" : "0 28px",
    },
    labelInline: {
      fontWeight: "700",
      marginRight: isMobile ? 6 : 14,
      fontSize: 17,
      color: "#59a3fc",
      display: "inline-block",
    },
    volumeDisplay: {
      fontWeight: "700",
      marginLeft: isMobile ? 6 : 10,
      color: "#93baff",
      fontSize: 15,
    },
    toggleSwitch: {
      marginTop: 18,
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      padding: isMobile ? "0 2vw" : "0 26px",
    },
    toggleLabel: {
      fontWeight: "700",
      fontSize: isMobile ? 15 : 17,
      color: "#59a3fc",
    },
    addCameraBtn: {
      marginTop: 24,
      background: "linear-gradient(135deg, #025dfe 0%, #161f38 100%)",
      color: "#fff",
      borderRadius: 24,
      padding: isMobile ? "11px 20vw" : "16px 48px",
      fontSize: isMobile ? 15 : 20,
      fontWeight: "800",
      cursor: "pointer",
      border: "none",
      boxShadow: "0 8px 27px #025dfe44",
      transition: "transform 0.3s, box-shadow 0.3s, background 0.3s",
      width: isMobile ? "98vw" : undefined
    },
    addCameraBtnHover: {
      transform: isMobile ? "none" : "translateY(-6px)",
      boxShadow: "0 16px 48px #025dfe88",
      background: "linear-gradient(135deg, #119aff 0%, #24375c 100%)",
    },
    deleteCameraBtn: {
      position: "absolute",
      top: isMobile ? 14 : 25,
      right: isMobile ? 8 : 18,
      background: "linear-gradient(135deg, #eb3349 0%, #89216b 100%)",
      border: "none",
      borderRadius: 8,
      color: "#fff",
      cursor: "pointer",
      padding: isMobile ? "6px 13px" : "7px 15px",
      fontWeight: "700",
      fontSize: isMobile ? 11 : 13,
      zIndex: 10,
      boxShadow: "0 4px 12px #eb334988",
      transition: "background 0.3s",
    },
    deleteCameraBtnHover: {
      background: "linear-gradient(135deg, #861657 0%, #eb3349 100%)",
    },
    buttonRow: {
      marginTop: isMobile ? 22 : 54,
      display: "flex",
      justifyContent: "center",
      gap: isMobile ? 11 : 36,
      flexWrap: "wrap",
      animation: "fadeInUp 1s",
    },
    actionButton: {
      background: "linear-gradient(135deg, #2962ff 0%, #161f38 100%)",
      color: "#fff",
      borderRadius: 22,
      padding: isMobile ? "12px 16vw" : "17px 46px",
      fontSize: isMobile ? 15 : 21,
      fontWeight: "900",
      cursor: "pointer",
      border: "none",
      boxShadow: "0 9px 32px #2962ff44",
      transition: "transform 0.3s, box-shadow 0.3s, background 0.3s",
      letterSpacing: '0.03em',
      marginBottom: isMobile ? 13 : 0,
      width: isMobile ? "98vw" : undefined
    },
    actionButtonHover: {
      transform: isMobile ? "none" : "translateY(-6px)",
      boxShadow: "0 18px 48px #1969fa99",
      background: "linear-gradient(135deg, #4e7cfa 0%, #101a2b 100%)",
    },
    alertPanel: {
      position: isMobile ? "static" : "fixed",
      top: isMobile ? undefined : 132,
      right: isMobile ? undefined : 28,
      width: isMobile ? "97vw" : 370,
      background: "linear-gradient(135deg, #1d2337 80%, #1756cb 100%)",
      boxShadow: "0 10px 44px #01152fbb",
      borderRadius: 26,
      padding: isMobile ? 12 : 26,
      zIndex: 90,
      animation: "fadeInUp 0.8s",
      color: "#eaf4fb",
      fontFamily: "'Poppins', 'Segoe UI'",
      border: "1.5px solid #274dc8",
      marginTop: isMobile ? 20 : 0,
      marginBottom: isMobile ? 17 : 0
    },
    alertTitle: {
      fontWeight: "900",
      fontSize: isMobile ? 16 : 24,
      marginBottom: isMobile ? 10 : 18,
      color: "#59a3fc",
      letterSpacing: '0.02em'
    },
    soundOption: {
      marginBottom: isMobile ? 9 : 14,
      fontSize: isMobile ? 13 : 17,
      fontWeight: "600",
      color: "#93baff",
      display: 'inline-block'
    },
    volumeInput: {
      width: "100%",
      height: 7,
      borderRadius: 6,
      background: "#002e53",
      cursor: "pointer",
      boxShadow: "0 2px 10px #59a3fc44"
    },
    cameraNameInput: {
      width: "100%",
      padding: isMobile ? 8 : 11,
      fontSize: isMobile ? 13 : 18,
      borderRadius: 12,
      border: "2px solid #284873",
      marginBottom: 12,
      background: "#122046",
      color: "#93baff",
      fontWeight: "700",
      transition: "border-color 0.3s, box-shadow 0.3s",
      boxShadow: "0 2px 14px #063e7277"
    },
    cameraNameInputFocus: {
      borderColor: "#59a3fc",
      background: "#13264d",
      boxShadow: "0 0 14px #3467b2dd",
      outline: "none",
    },
    "@keyframes fadeInDown": {
      from: { opacity: 0, transform: "translateY(-30px)" },
      to: { opacity: 1, transform: "translateY(0)" },
    },
    "@keyframes fadeInUp": {
      from: { opacity: 0, transform: "translateY(40px)" },
      to: { opacity: 1, transform: "translateY(0)" },
    },
    "@keyframes fadeIn": {
      from: { opacity: 0 },
      to: { opacity: 1 },
    },
    "@keyframes slideInLeft": {
      from: { opacity: 0, transform: "translateX(-60px)" },
      to: { opacity: 1, transform: "translateX(0)" },
    },
  };



  // Play sound effect for 3 seconds when alert sound changes
  const audioRef = useRef(null); // ref to audio element

  const [cameraPairs, setCameraPairs] = useState([
    {
      pairId: 0, // locked base cameras, no delete allowed
      cameras: [
        {
          id: 0,
          name: "Camera 0 - Sample Video",
          src: "/people.mp4", // place sample.video in public/videos for serve
          on: true,
          alertSound: "sound1",
          volume: 50,
          uploadedFile: null,
        },
        {
          id: 1,
          name: "Camera 0b - Output Video",
          src: "/output_with_heatmap.gif",
          on: true,
          alertSound: "sound1",
          volume: 50,
          uploadedFile: null,
        },
      ],
    },
  ]);

  const [availableCameras] = useState([
    "Integrated Webcam",
    "USB Camera 1",
    "USB Camera 2",
    "Virtual Camera",
  ]);


  useEffect(() => {
    // Handle outside clicks for all dropdown/panels
    function handleClickOutside(event) {
      if (accountRef.current && !accountRef.current.contains(event.target) && accountOpen)
        setAccountOpen(false);
      if (alertRef.current && !alertRef.current.contains(event.target) && alertOpen)
        setAlertOpen(false);
      if (systemSettingsRef.current && !systemSettingsRef.current.contains(event.target) && systemSettingsOpen)
        setSystemSettingsOpen(false);
    }

    document.addEventListener("mousedown", handleClickOutside);

    // Play or pause alert sound when alertOpen / selectedSound / volume change
    if (alertOpen && audioRef.current) {
      audioRef.current.src = `/sounds/${selectedSound}.mp3`;
      audioRef.current.volume = volume / 100;
      audioRef.current.loop = true;
      audioRef.current.load();
      audioRef.current.play().catch(e => {
        console.log("Alert sound play prevented:", e);
      });
    } else if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }

    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [accountOpen, alertOpen, systemSettingsOpen, selectedSound, volume]);

  const toggleCamera = (pairId, cameraId) => {
    setCameraPairs((pairs) =>
      pairs.map((pair) =>
        pair.pairId === pairId
          ? {
            ...pair,
            cameras: pair.cameras.map((cam) => (cam.id === cameraId ? { ...cam, on: !cam.on } : cam)),
          }
          : pair
      )
    );
  };

  const updateCameraName = (pairId, cameraId, newName) => {
    setCameraPairs((pairs) =>
      pairs.map((pair) =>
        pair.pairId === pairId
          ? {
            ...pair,
            cameras: pair.cameras.map((cam) => (cam.id === cameraId ? { ...cam, name: newName } : cam)),
          }
          : pair
      )
    );
  };

  // Toggle alert sound per camera
  const toggleAlertSound = (pairId, cameraId) => {
    setCameraPairs(pairs =>
      pairs.map(pair =>
        pair.pairId === pairId
          ? {
            ...pair,
            cameras: pair.cameras.map(cam =>
              cam.id === cameraId ? { ...cam, alertSound: !cam.alertSound } : cam
            ),
          }
          : pair
      )
    );
  };

  const setCameraAlertSound = (pairId, cameraId, sound) => {
    setCameraPairs((pairs) =>
      pairs.map((pair) =>
        pair.pairId === pairId
          ? {
            ...pair,
            cameras: pair.cameras.map((cam) => (cam.id === cameraId ? { ...cam, alertSound: sound } : cam)),
          }
          : pair
      )
    );
    playSound(sound);
  };

  const changeVolume = (pairId, cameraId, vol) => {
    setCameraPairs((pairs) =>
      pairs.map((pair) =>
        pair.pairId === pairId
          ? {
            ...pair,
            cameras: pair.cameras.map((cam) => (cam.id === cameraId ? { ...cam, volume: vol } : cam)),
          }
          : pair
      )
    );
  };

  const uploadVideoFile = async (pairId, cameraId, file) => {
    // Show loading state if you wish

    const formData = new FormData();
    formData.append("video", file);

    // Send video to FastAPI backend
    const res = await fetch("http://http://0.0.0.0:8000/process_video/", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    // Set input frame src to uploaded video (local, instant)
    // Set output frame src to backend returned output video URL
    // Tip: You may want to use /download_output?path=... for file serving in production

    setCameraPairs(pairs =>
      pairs.map(pair =>
        pair.pairId === pairId
          ? {
            ...pair,
            cameras: pair.cameras.map(cam =>
              cam.id === cameraId
                ? { ...cam, src: URL.createObjectURL(file), uploadedFile: file }
                : cam.id === cameraId + 1 // Assuming output cam is always id+1
                  ? { ...cam, src: `http://http://0.0.0.0:8000/download_output?path=${encodeURIComponent(data.output_video)}` }
                  : cam
            ),
          }
          : pair
      )
    );

    // Optionally fetch crowd.txt file and show result
    fetch("http://http://0.0.0.0:8000/crowd_txt/")
      .then(res => res.text())
      .then(txt => {
        // You can store this in a state and show under the video card
        // setCrowdAnalytics(txt);
        // Example: setCrowdTxtMap({ ...crowdTxtMap, [pairId]: txt });
      }).catch(() => { /* handle error */ });
  };





  const playSound = (sound) => {
    if (audioRef.current) {
      audioRef.current.src = `/sounds/${sound}.mp3`;
      audioRef.current.load();
      audioRef.current.volume = 1;
      audioRef.current.play().catch((e) => {
        console.log("Audio playback failed:", e);
      });
      setTimeout(() => {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
      }, 3000);
    }
  };

  // Add new camera pair
  const addCameraPair = () => {
    const newPairId = cameraPairs.length ? cameraPairs[cameraPairs.length - 1].pairId + 1 : 1;
    const baseCamId = cameraPairs.reduce((max, pair) => Math.max(max, ...pair.cameras.map(c => c.id)), 0);

    setCameraPairs(prev => [
      ...prev,
      {
        pairId: newPairId,
        cameras: [
          {
            id: baseCamId + 1,
            name: `Camera ${newPairId} - Real`,
            src: "https://placeimg.com/640/480/nature",
            on: true,
            alertSound: false,
            volume: 50,
            uploadedFile: null,
          },
          {
            id: baseCamId + 2,
            name: `Camera ${newPairId}b - Model Video`,
            src: "https://placeimg.com/640/480/tech",
            on: true,
            alertSound: false,
            volume: 50,
            uploadedFile: null,
          },
        ],
      },
    ]);
  };

  // Delete camera pair except Camera 0 (pairId 0 locked)
  const deleteCameraPair = (pairId) => {
    if (pairId === 0) return; // prevent deletion of Camera 0
    setCameraPairs(pairs => pairs.filter(pair => pair.pairId !== pairId));
  };


  // Add selected system camera as new camera pair
  const addSystemCamera = () => {
    if (!selectedSystemCamera) return;
    const newPairId = cameraPairs.length ? cameraPairs[cameraPairs.length - 1].pairId + 1 : 1;
    setCameraPairs(prev => [
      ...prev,
      {
        pairId: newPairId,
        cameras: [
          {
            id: newPairId * 2 - 1,
            name: `${selectedSystemCamera} - Real`,
            src: "https://placeimg.com/640/480/tech",
            on: true,
          },
          {
            id: newPairId * 2,
            name: `${selectedSystemCamera} - Model Video`,
            src: "https://placeimg.com/640/480/arch",
            on: true,
          },
        ],
      },
    ]);
    setSelectedSystemCamera("");
    setSystemSettingsOpen(false);
  };

  // Add uploaded video as new camera pair
  const addUploadedVideo = () => {
    if (!uploadedVideoURL) return;
    const newPairId = cameraPairs.length ? cameraPairs[cameraPairs.length - 1].pairId + 1 : 1;
    setCameraPairs(prev => [
      ...prev,
      {
        pairId: newPairId,
        cameras: [
          {
            id: newPairId * 2 - 1,
            name: `Uploaded Video ${newPairId} - Real`,
            src: uploadedVideoURL,
            on: true,
          },
          {
            id: newPairId * 2,
            name: `Uploaded Video ${newPairId}b - Model Video`,
            src: uploadedVideoURL,
            on: true,
          },
        ],
      },
    ]);
    setUploadedVideoURL(null);
    setSystemSettingsOpen(false);
  };

  const welcomeName = user?.role === "admin" ? user?.departmentName : user?.name;

  return (
    <div style={styles.page}>

      {/* Welcome and Account */}
      <div style={styles.welcomeRow}>
        <div style={styles.welcomeText}>Welcome, {welcomeName || "User"}! Role: {user?.role || "N/A"}</div>


        <div
          style={styles.accountType}
          onClick={() => setAccountOpen(v => !v)}
          title="Click to view account details"
          ref={accountRef}
        >
          Account ▼
          {accountOpen && (
            <div style={styles.accountDetailsDropdown}>
              <div><span style={styles.label}>Role: </span>{user?.role || "N/A"}</div>
              <div><span style={styles.label}>Email: </span>{user?.email || "N/A"}</div>
              <div><span style={styles.label}>Account Details: </span>{user?.accountDetails || "No info"}</div>
            </div>
          )}
        </div>
      </div>

      {/* Camera pairs */}
      <div style={styles.cameraPairsContainer}>
        {cameraPairs.map(pair => {
          const isCamera0 = pair.pairId === 0;

          return (
            <div key={pair.pairId} style={styles.cameraPair}>
              {pair.cameras.map((cam) => {
                const isModelVideo = cam.name.toLowerCase().includes("b"); // model video if name has 'b'

                return (
                  <div key={cam.id} style={styles.cameraCard}>
                    <div style={styles.cameraTitle}>
                      <input
                        type="text"
                        value={cam.name}
                        onChange={e => updateCameraName(pair.pairId, cam.id, e.target.value)}
                        style={styles.cameraNameInput}
                        disabled={isCamera0}
                      />
                    </div>
                    <div style={styles.cameraFrame}>
                      {cam.on ? (
                        <>
                          {cam.src.endsWith(".mp4") || cam.src.endsWith(".webm") ? (
                            <video autoPlay muted loop style={styles.cameraVideo} src={cam.src} />
                          ) : (
                            <img style={styles.cameraVideo} alt={`${cam.name} Feed`} src={cam.src} />
                          )}
                        </>
                      ) : (
                        <div style={{ color: "#888", fontStyle: "italic" }}>Camera Off</div>
                      )}
                    </div>

                    <div style={styles.toggleSwitch}>
                      <label style={styles.toggleLabel}>{cam.name} On/Off</label>
                      <input
                        type="checkbox"
                        checked={cam.on}
                        onChange={() => toggleCamera(pair.pairId, cam.id)}
                      />
                    </div>

                    {isModelVideo && !isCamera0 && (
                      <>
                        <div style={styles.cameraSettings}>
                          <span style={{ fontWeight: "600", marginRight: 8 }}>Alert Sound:</span>
                          {["sound1", "sound2", "sound3"].map(sound => (
                            <label
                              key={sound}
                              style={{ marginRight: 12, cursor: "pointer" }}
                            >
                              <input
                                type="radio"
                                name={`alertSound-${pair.pairId}-${cam.id}`}
                                value={sound}
                                checked={cam.alertSound === sound}
                                onChange={() => setCameraAlertSound(pair.pairId, cam.id, sound)}
                              />
                              {sound.charAt(sound.length - 1).toUpperCase() + sound.slice(1)}
                              <audio ref={audioRef} />
                            </label>
                          ))}
                        </div>



                        <div style={styles.cameraSettings}>
                          <span style={{ fontWeight: "600", marginRight: 8 }}>Volume:</span>
                          <input
                            type="range"
                            min="0"
                            max="100"
                            value={cam.volume}
                            onChange={e => changeVolume(pair.pairId, cam.id, +e.target.value)}
                          />
                          <span style={styles.volumeDisplay}>{cam.volume}%</span>
                        </div>
                      </>
                    )}



                    {/* Upload video only on real video and not Camera 0 */}
                    {!isModelVideo && !isCamera0 && (
                      <div style={styles.cameraSettings}>
                        <label>
                          Upload Video:
                          <input
                            type="file"
                            accept="video/*"
                            onChange={e => {
                              if (e.target.files && e.target.files[0]) {
                                uploadVideoFile(pair.pairId, cam.id, e.target.files[0]);
                              }
                            }}
                          />
                        </label>
                      </div>
                    )}

                    {/* Delete button for first camera only and not on Camera 0 */}
                    {!isCamera0 && cam.id === pair.cameras[0].id && (
                      <button
                        onClick={() => deleteCameraPair(pair.pairId)}
                        style={styles.deleteCameraBtn}
                        title="Delete this camera pair"
                      >
                        Delete
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>

      {/* Hidden audio element */}
      <audio ref={audioRef} />


      <button style={styles.addCameraBtn} onClick={addCameraPair}>
        + Add More Camera Pair
      </button>

      {/* Action buttons */}
      <div style={styles.buttonRow}>
        <button
          style={styles.actionButton}
          onClick={() => setSystemSettingsOpen(v => !v)}
        >
          System Settings
        </button>


        <button
          style={styles.actionButton}
          onClick={() => navigate("/dashboard")}
        >
          Analytics Dashboard
        </button>

        <button
          style={styles.actionButton}
          onClick={() => navigate("/crowd_count_photo")}
        >
          Crowd Counting from Photo
        </button>

        <button
          style={styles.actionButton}
          onClick={() => setAlertOpen((v) => !v)}
        >
          View Alerts
        </button>
        <audio ref={audioRef} />
      </div>

      {/* Alert panel */}
      {alertOpen && (
        <div style={styles.alertPanel} ref={alertRef}>
          <div style={styles.alertTitle}>Alert Sound Settings</div>

          <div>
            <input type="radio" id="sound1" name="globalAlertSound" value="sound1" checked={selectedSound === "sound1"} onChange={() => setSelectedSound("sound1")} />
            <label htmlFor="sound1">Sound 1</label>
          </div>

          <div>
            <input type="radio" id="sound2" name="globalAlertSound" value="sound2" checked={selectedSound === "sound2"} onChange={() => setSelectedSound("sound2")} />
            <label htmlFor="sound2">Sound 2</label>
          </div>

          <div>
            <input type="radio" id="sound3" name="globalAlertSound" value="sound3" checked={selectedSound === "sound3"} onChange={() => setSelectedSound("sound3")} />
            <label htmlFor="sound3">Sound 3</label>
          </div>

          <label htmlFor="volume" style={{ fontWeight: "700" }}>
            Volume: {volume}%
          </label>
          <input type="range" id="volume" min="0" max="100" value={volume} onChange={(e) => setVolume(+e.target.value)} style={styles.volumeInput} />
        </div>
      )}

      {/* System Settings panel */}
      {systemSettingsOpen && (
        <div style={styles.alertPanel} ref={systemSettingsRef}>
          {/* <h3 style={{ marginBottom: 12, color: "#0a2647" }}>System Settings Coming Soon</h3> */}

          {/* Camera selection */}
          <label style={{ fontWeight: "700" }}>Select Camera:</label>
          <select
            value={selectedSystemCamera}
            onChange={e => setSelectedSystemCamera(e.target.value)}
            style={{ width: "100%", padding: 8, marginBottom: 16 }}
          >
            <option value="">-- Choose a camera --</option>
            {availableCameras.map((cam, idx) => (
              <option key={idx} value={cam}>{cam}</option>
            ))}
          </select>
          <button
            disabled={!selectedSystemCamera}
            onClick={addSystemCamera}
            style={{
              ...styles.actionButton,
              marginBottom: 20,
              width: "100%",
              opacity: selectedSystemCamera ? 1 : 0.5,
            }}
          >
            Add Selected Camera
          </button>

          <button
            onClick={() => setSystemSettingsOpen(false)}
            style={{ marginTop: 12, width: "100%", backgroundColor: "#ccc", color: "#333" }}
          >
            Close Settings
          </button>

          {/* Video Upload */}
          <label style={{ fontWeight: "700" }}>Upload Video:</label>
          <input
            type="file"
            accept="video/*"
            onChange={e => {
              if (e.target.files && e.target.files[0]) {
                const fileURL = URL.createObjectURL(e.target.files[0]);
                setUploadedVideoURL(fileURL);
              }
            }}
            style={{ marginBottom: 12 }}
          />
          {uploadedVideoURL && (
            <>
              <video
                src={uploadedVideoURL}
                controls
                style={{ width: "100%", borderRadius: 12, marginBottom: 12 }}
              />
              <button
                onClick={addUploadedVideo}
                style={{ ...styles.actionButton, width: "100%" }}
              >
                Add Uploaded Video as Camera Pair
              </button>
            </>
          )}

          <button
            onClick={() => setSystemSettingsOpen(false)}
            style={{ marginTop: 12, width: "100%", backgroundColor: "#ccc", color: "#333" }}
          >
            Close Settings
          </button>
        </div>
      )}
    </div>
  );
}

export default MainPage;

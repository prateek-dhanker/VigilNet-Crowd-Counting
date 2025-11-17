// @app.route('/api/signup', methods=['POST'])
// def signup():
//     data = request.json
//     conn = None
//     cursor = None

//     try:
//         conn = mysql.connector.connect(**db_config)
//         cursor = conn.cursor()

//         # Validate role-based unique key decisions
//         role = data.get('role')
//         if role == 'admin':
//             primary_key_field = 'dept_id'
//             primary_key_value = data.get('deptId')
//             if not primary_key_value:
//                 return jsonify({"success": False, "message": "Department ID is required for admin."}), 400
//         elif role == 'user':
//             primary_key_field = 'aadhar'
//             primary_key_value = data.get('aadhar')
//             if not primary_key_value:
//                 return jsonify({"success": False, "message": "Aadhar Card number is required for user."}), 400
//         else:
//             return jsonify({"success": False, "message": "Invalid role provided."}), 400

//         sql = """
//             INSERT INTO users (
//                 role, first_name, middle_name, last_name, dob, gender, dept_name, dept_id, aadhar,
//                 address, state, country, email, password, mobile, alt_mobile, security_question,
//                 security_answer
//             ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
//         """

//         values = (
//             role,
//             data.get('firstName'),
//             data.get('middleName'),
//             data.get('lastName'),
//             data.get('dob'),
//             data.get('gender'),
//             data.get('deptName') if role == 'admin' else None,
//             data.get('deptId') if role == 'admin' else None,
//             data.get('aadhar') if role == 'user' else None,
//             data.get('address'),
//             data.get('state'),
//             data.get('country'),
//             data.get('email'),
//             hash_password(data.get('password')) if data.get('password') else None,
//             data.get('mobile'),
//             data.get('altMobile'),
//             data.get('securityQuestion'),
//             data.get('securityAnswer')
//         )

//         cursor.execute(sql, values)
//         conn.commit()

//         return jsonify({"success": True, "message": "Signup successful."}), 200

//     except mysql.connector.IntegrityError as ie:
//         print("IntegrityError:", ie)
//         err_msg = str(ie).lower()
//         if 'dept_id' in err_msg or 'aadhar' in err_msg:
//             return jsonify({"success": False, "message": f"{primary_key_field.replace('_', ' ').title()} already exists."}), 400
//         elif 'email' in err_msg:
//             return jsonify({"success": False, "message": "Email already exists."}), 400
//         return jsonify({"success": False, "message": "Database integrity error."}), 400

//     except Exception as e:
//         print("Error during signup:", e)
//         return jsonify({"success": False, "message": f"Signup failed: {str(e)}"}), 500

//     finally:
//         if cursor:
//             cursor.close()
//         if conn:
//             conn.close()




import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";

const styles = {
  page: {
    background: "linear-gradient(135deg, #74ebd5 0%, #ACB6E5 100%)",
    minHeight: "100vh",
    padding: "40px 60px",
    fontFamily: "'Poppins', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    color: "#222",
    display: "flex",
    flexDirection: "column",
    gap: 24,
    position: "relative",
  },
  welcomeRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 24,
  },
  welcomeText: {
    fontSize: 22,
    fontWeight: "600",
    color: "#093554",
  },
  accountType: {
    position: "relative",
    cursor: "pointer",
    padding: "8px 16px",
    backgroundColor: "#0a84ff",
    color: "#fff",
    borderRadius: 12,
    userSelect: "none",
  },
  accountDetailsDropdown: {
    position: "absolute",
    top: "110%",
    right: 0,
    background: "#fff",
    boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
    borderRadius: 12,
    padding: 16,
    width: 280,
    zIndex: 10,
  },
  label: {
    fontWeight: "600",
    color: "#0a2647",
  },
  cameraPairsContainer: {
    display: "flex",
    flexDirection: "column",
    gap: 24,
    flexWrap: "nowrap",
  },
  cameraPair: {
    display: "flex",
    gap: 20,
    alignItems: "flex-start",
  },
  cameraCard: {
    background: "#fff",
    borderRadius: 16,
    boxShadow: "0 8px 24px rgb(0 0 0 / 0.12)",
    overflow: "hidden",
    paddingBottom: 16,
    animation: "fadeInUp 1s ease forwards",
    flex: "1 1 45%",
    minWidth: 320,
    position: "relative",
  },
  cameraTitle: {
    fontSize: 20,
    fontWeight: "600",
    padding: "16px 20px 8px",
    color: "#0a2647",
  },
  cameraFrame: {
    width: "100%",
    height: 240,
    backgroundColor: "#000",
    borderRadius: 12,
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    overflow: "hidden",
    position: "relative",
  },
  cameraImg: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
  },
  toggleSwitch: {
    marginTop: 12,
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "0 20px",
  },
  toggleLabel: {
    fontWeight: "600",
    fontSize: 16,
    color: "#0a2647",
  },
  addCameraBtn: {
    marginTop: 20,
    background: "linear-gradient(135deg, #16c79a 0%, #087f55 100%)",
    color: "#fff",
    borderRadius: 12,
    padding: "14px 32px",
    fontSize: 18,
    fontWeight: "700",
    cursor: "pointer",
    border: "none",
    boxShadow: "0 6px 20px #04502e88",
    transition: "transform 0.3s ease, box-shadow 0.3s ease",
    width: 220,
  },
  deleteCameraBtn: {
    position: "absolute",
    top: 10,
    right: 10,
    background: "#e03e3e",
    border: "none",
    borderRadius: 6,
    color: "#fff",
    cursor: "pointer",
    padding: "4px 8px",
    fontWeight: "700",
    fontSize: 12,
    zIndex: 5,
  },
  buttonRow: {
    marginTop: 40,
    display: "flex",
    justifyContent: "center",
    gap: 30,
    flexWrap: "wrap",
  },
  actionButton: {
    background: "linear-gradient(135deg, #0a84ff 0%, #0051a2 100%)",
    color: "#fff",
    borderRadius: 12,
    padding: "14px 32px",
    fontSize: 18,
    fontWeight: "700",
    cursor: "pointer",
    border: "none",
    boxShadow: "0 6px 20px #00408088",
    transition: "transform 0.3s ease, box-shadow 0.3s ease",
  },
  alertPanel: {
    position: "fixed",
    top: 100,
    right: 20,
    width: 320,
    background: "#fff",
    boxShadow: "0 6px 30px rgba(0,0,0,0.3)",
    borderRadius: 16,
    padding: 20,
    zIndex: 50,
  },
  alertTitle: {
    fontWeight: "700",
    fontSize: 20,
    marginBottom: 12,
    color: "#0a2647",
  },
  soundOption: {
    marginBottom: 12,
  },
  volumeInput: {
    width: "100%",
  },
  cameraSettings: {
    marginTop: 12,
    padding: "0 20px",
  },
  cameraNameInput: {
    width: "100%",
    padding: 6,
    fontSize: 16,
    borderRadius: 6,
    border: "1px solid #ccc",
    marginBottom: 8,
  },
  labelInline: {
    fontWeight: "600",
    marginRight: 8,
  }
};

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

  // const [cameraPairs, setCameraPairs] = useState([
  //   {
  //     pairId: 1,
  //     cameras: [
  //       { id: 1, name: "Camera 1 - Real", src: "https://placeimg.com/640/480/tech", on: true },
  //       { id: 2, name: "Camera 1b - Model Video", src: "https://placeimg.com/640/480/arch", on: true },
  //     ],
  //   },
  // ]);

  const [cameraPairs, setCameraPairs] = useState([
    {
      pairId: 0, // locked base cameras, no delete allowed
      cameras: [
        {
          id: 0,
          name: "Camera 0 - Sample Video",
          src: "/videos/sample.video", // place sample.video in public/videos for serve
          on: true,
          alertSound: false,
          volume: 50,
          uploadedFile: null,
        },
        {
          id: 1,
          name: "Camera 0b - Output Video",
          src: "/videos/sample_output_video",
          on: true,
          alertSound: false,
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
  



  // Close dropdowns/panels on outside clicks
  // useEffect(() => {
  //   function handleClickOutside(event) {
  //     if (accountRef.current && !accountRef.current.contains(event.target) && accountOpen) {
  //       setAccountOpen(false);
  //     }
  //     if (alertRef.current && !alertRef.current.contains(event.target) && alertOpen) {
  //       setAlertOpen(false);
  //     }
  //     if (systemSettingsRef.current && !systemSettingsRef.current.contains(event.target) && systemSettingsOpen) {
  //       setSystemSettingsOpen(false);
  //       setSelectedSystemCamera("");
  //       setUploadedVideoURL(null);
  //     }
  //   }
  //   document.addEventListener("mousedown", handleClickOutside);
  //   return () => document.removeEventListener("mousedown", handleClickOutside);
  // }, [accountOpen, alertOpen, systemSettingsOpen]);

  useEffect(() => {
    function handleClickOutside(event) {
      if (accountRef.current && !accountRef.current.contains(event.target) && accountOpen) setAccountOpen(false);
      if (alertRef.current && !alertRef.current.contains(event.target) && alertOpen) setAlertOpen(false);
      if (systemSettingsRef.current && !systemSettingsRef.current.contains(event.target) && systemSettingsOpen) setSystemSettingsOpen(false);
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [accountOpen, alertOpen, systemSettingsOpen]);

  // const toggleCamera = (pairId, cameraId) => {
  //   setCameraPairs(prevPairs =>
  //     prevPairs.map(pair =>
  //       pair.pairId === pairId
  //         ? { ...pair, cameras: pair.cameras.map(cam => cam.id === cameraId ? { ...cam, on: !cam.on } : cam) }
  //         : pair
  //     )
  //   );
  // };

  // Toggle camera on/off
  const toggleCamera = (pairId, cameraId) => {
    setCameraPairs(pairs =>
      pairs.map(pair => {
        if (pair.pairId === pairId) {
          return {
            ...pair,
            cameras: pair.cameras.map(cam =>
              cam.id === cameraId ? { ...cam, on: !cam.on } : cam
            ),
          };
        }
        return pair;
      })
    );
  };

  // Update camera name individually
  const updateCameraName = (pairId, cameraId, newName) => {
    setCameraPairs(pairs =>
      pairs.map(pair =>
        pair.pairId === pairId
          ? {
            ...pair,
            cameras: pair.cameras.map(cam =>
              cam.id === cameraId ? { ...cam, name: newName } : cam
            ),
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

  // Change volume per camera
  const changeVolume = (pairId, cameraId, vol) => {
    setCameraPairs(pairs =>
      pairs.map(pair =>
        pair.pairId === pairId
          ? {
            ...pair,
            cameras: pair.cameras.map(cam =>
              cam.id === cameraId ? { ...cam, volume: vol } : cam
            ),
          }
          : pair
      )
    );
  };

  // Upload video file per camera to replace src
  const uploadVideoFile = (pairId, cameraId, file) => {
    const fileURL = URL.createObjectURL(file);
    setCameraPairs(pairs =>
      pairs.map(pair =>
        pair.pairId === pairId
          ? {
            ...pair,
            cameras: pair.cameras.map(cam =>
              cam.id === cameraId
                ? { ...cam, src: fileURL, uploadedFile: file }
                : cam
            ),
          }
          : pair
      )
    );
  };


  // const addCameraPair = () => {
  //   const newPairId = cameraPairs.length ? cameraPairs[cameraPairs.length - 1].pairId + 1 : 1;
  //   const baseCamId = cameraPairs.reduce((max, pair) => {
  //     const maxCam = Math.max(...pair.cameras.map(c => c.id));
  //     return maxCam > max ? maxCam : max;
  //   }, 0);

  //   setCameraPairs(prevPairs => [
  //     ...prevPairs,
  //     {
  //       pairId: newPairId,
  //       cameras: [
  //         {
  //           id: baseCamId + 1,
  //           name: `Camera ${newPairId} - Real`,
  //           src: "https://placeimg.com/640/480/nature",
  //           on: true,
  //         },
  //         {
  //           id: baseCamId + 2,
  //           name: `Camera ${newPairId}b - Model Video`,
  //           src: "https://placeimg.com/640/480/tech",
  //           on: true,
  //         },
  //       ],
  //     },
  //   ]);
  // };

  // const deleteCameraPair = (pairId) => {
  //   setCameraPairs(prevPairs => prevPairs.filter(pair => pair.pairId !== pairId));
  // };



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

      {/* Camera pairs display
      <div style={styles.cameraPairsContainer}>
        {cameraPairs.map(pair => (
          <div key={pair.pairId} style={styles.cameraPair}>
            {pair.cameras.map(cam => (
              <div key={cam.id} style={styles.cameraCard}>
                <div style={styles.cameraTitle}>{cam.name}</div>
                <div style={styles.cameraFrame}>
                  {cam.on ? (
                    <img style={styles.cameraImg} alt={`${cam.name} Feed`} src={cam.src} />
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
                {cam.id === pair.cameras[0].id && (
                  <button
                    onClick={() => deleteCameraPair(pair.pairId)}
                    style={styles.deleteCameraBtn}
                    title="Delete this camera pair"
                  >
                    Delete
                  </button>
                )}
              </div>
            ))}
          </div>
        ))}
      </div>

      <button style={styles.addCameraBtn} onClick={addCameraPair}>
        + Add More Camera Pair
      </button> */}

      {/* Camera pairs */}
      <div style={styles.cameraPairsContainer}>
        {cameraPairs.map(pair => (
          <div key={pair.pairId} style={styles.cameraPair}>

            {pair.cameras.map(cam => (
              <div key={cam.id} style={styles.cameraCard}>
                <div style={styles.cameraTitle}>
                  <input
                    type="text"
                    value={cam.name}
                    onChange={e => updateCameraName(pair.pairId, cam.id, e.target.value)}
                    style={styles.cameraNameInput}
                    disabled={pair.pairId === 0} // disable editing Camera 0 name
                  />
                </div>
                <div style={styles.cameraFrame}>
                  {cam.on ? (
                    <>
                      {cam.src.endsWith(".mp4") || cam.src.endsWith(".webm") ? (
                        <video autoPlay muted loop style={styles.cameraImg} src={cam.src} />
                      ) : (
                        <img style={styles.cameraImg} alt={`${cam.name} Feed`} src={cam.src} />
                      )}
                    </>
                  ) : (
                    <div style={{ color: "#888", fontStyle: "italic" }}>Camera Off</div>
                  )}
                </div>
                <div style={styles.cameraSettings}>
                  <label style={styles.labelInline}>
                    Alert Sound:
                    <input
                      type="checkbox"
                      checked={cam.alertSound}
                      onChange={() => toggleAlertSound(pair.pairId, cam.id)}
                      disabled={pair.pairId === 0} // lock alert sound on Camera 0
                    />
                  </label>
                  <label style={styles.labelInline}>
                    Volume:
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={cam.volume}
                      onChange={e => changeVolume(pair.pairId, cam.id, +e.target.value)}
                      disabled={pair.pairId === 0} // lock volume on Camera 0
                    />
                  </label>
                </div>

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
                      disabled={pair.pairId === 0} // disable on Camera 0
                    />
                  </label>
                </div>

                {pair.pairId !== 0 && cam.id === pair.cameras[0].id && (
                  <button
                    onClick={() => deleteCameraPair(pair.pairId)}
                    style={styles.deleteCameraBtn}
                    title="Delete this camera pair"
                  >
                    Delete
                  </button>
                )}

                <div style={styles.toggleSwitch}>
                  <label style={styles.toggleLabel}>{cam.name} On/Off</label>
                  <input
                    type="checkbox"
                    checked={cam.on}
                    onChange={() => toggleCamera(pair.pairId, cam.id)}
                  />
                </div>
              </div>
            ))}

          </div>
        ))}
      </div>

      <button style={styles.addCameraBtn} onClick={addCameraPair}>
        + Add More Camera Pair
      </button>

      {/* Action buttons
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
          onClick={() => setAlertOpen(v => !v)}
        >
          View Alerts
        </button>
      </div> */}


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
          onClick={() => navigate("/crowd-count-photo")}
        >
          Crowd Counting from Photo
        </button>

        <button
          style={styles.actionButton}
          onClick={() => setAlertOpen(v => !v)}
        >
          View Alerts
        </button>
      </div>

      {/* Alert panel
      {alertOpen && (
        <div style={styles.alertPanel} ref={alertRef}>
          <div style={styles.alertTitle}>Alert Sound Settings</div>

          <div style={styles.soundOption}>
            <input
              type="radio"
              id="sound1"
              name="alertSound"
              value="sound1"
              checked={selectedSound === "sound1"}
              onChange={e => setSelectedSound(e.target.value)}
            />
            <label htmlFor="sound1">Sound 1</label>
          </div>

          <div style={styles.soundOption}>
            <input
              type="radio"
              id="sound2"
              name="alertSound"
              value="sound2"
              checked={selectedSound === "sound2"}
              onChange={e => setSelectedSound(e.target.value)}
            />
            <label htmlFor="sound2">Sound 2</label>
          </div>

          <div style={styles.soundOption}>
            <input
              type="radio"
              id="sound3"
              name="alertSound"
              value="sound3"
              checked={selectedSound === "sound3"}
              onChange={e => setSelectedSound(e.target.value)}
            />
            <label htmlFor="sound3">Sound 3</label>
          </div>

          <label htmlFor="volume" style={{ fontWeight: "700" }}>
            Volume: {volume}
          </label>
          <input
            type="range"
            id="volume"
            min="0"
            max="100"
            value={volume}
            onChange={e => setVolume(parseInt(e.target.value, 10))}
            style={styles.volumeInput}
          />
        </div>
      )} */}


      

      {/* System Settings panel */}
      {systemSettingsOpen && (
        // <div style={styles.alertPanel} ref={systemSettingsRef}>
        //   <h3 style={{ marginBottom: 12, color: "#0a2647" }}>System Settings</h3>
        <div style={styles.alertPanel} ref={systemSettingsRef}>
          <h3 style={{ marginBottom: 12, color: "#0a2647" }}>System Settings Coming Soon</h3>

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

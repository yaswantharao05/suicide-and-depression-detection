import { useState } from "react";
import axios from "axios";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import "./styles.css";

export default function App() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState([]);
  const [frequentWords, setFrequentWords] = useState([]);
  const [message, setMessage] = useState([]);
  const [predicted_class, setPredicted_class] = useState([]);
  const [loading, setLoading] = useState(false); 
  const [showResults, setShowResults] = useState(false);

  const handleFileChange = (e) => setFile(e.target.files[0]);

  const handleUpload = async () => {
    if (!file) return alert("Please select a CSV file.");
  
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
  
    try {
      console.log("Stage 1: Sending request to backend...");
      const response = await axios.post(
        "http://127.0.0.1:8000/upload/", 
        formData, 
        {
          headers: { 
            "Content-Type": "multipart/form-data",
          },
        }
      );
      console.log("Stage 2: Received response:", response.data);
      
      setResults(response.data.flagged_texts || []);
      setFrequentWords(response.data.frequent_words || []);
      setMessage(response.data.message || []);
      setPredicted_class(response.data.predicted_class || []);
      
      console.log("Stage 3: State updated successfully");
    } catch (error) {
      console.error("Upload error:", error);
      
      let errorMessage = "Error processing file. ";
      if (error.response) {
        // The request was made and the server responded with a status code
        errorMessage += `Server responded with ${error.response.status}: ${error.response.data?.detail || 'No details'}`;
      } else if (error.request) {
        // The request was made but no response was received
        errorMessage += "No response from server. Check if backend is running." + error;
      } else {
        // Something happened in setting up the request
        errorMessage += error.message;
      }
      
      alert(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleShowResults = () => {
    setShowResults(!showResults);
  };

  return (
    <div className="container">
      <h1 className="heading">🧠 Early Suicide & Depression Detection</h1>

      <input
        type="file"
        accept=".csv"
        onChange={handleFileChange}
        className="file-input"
      />
      <button onClick={handleUpload} disabled={loading} className="upload-button">
        {loading ? "Processing..." : "Upload & Analyze"}
      </button>

      {loading && (
        <div className="loading-screen">
          <div className="spinner"></div>
          <p>Analyzing your data...</p>
        </div>
      )}

      {results.length > 0 && (
        <div className="results-container">
          <h2 className="text-xl font-semibold mb-4">Flagged Messages 🚨</h2>
          <table className="results-table">
            <thead>
              <tr>
                <th>Text (Preview)</th>
                <th>Suicidal Probability</th>
              </tr>
            </thead>
            <tbody>
              {results.map((item, index) => (
                <tr key={index}>
                  <td>{item.text.substring(0, 100)}...</td>
                  <td>{(item.probability * 100).toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="chart-container">
            <h2 className="chart-title">Most Frequent Suicide-Related Words</h2>
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={frequentWords.map(([word, count]) => ({ word, count }))}>
                <XAxis dataKey="word" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>

            <br/> <br/> <br/>
            <button onClick={handleShowResults} className="upload-button">
              Show Analysis Results
            </button>
            
            <br />
            {showResults && (
              <div className="prediction-results">
                <h2>System Prediction: <span className="predicted-class">{predicted_class[0]}</span></h2>
                <div className="message-container">
                  <h3>Recommendations:</h3>
                  <div className="message-content">
                    {message.map((point, index) => (
                      <div key={index} className="message-line">
                        {index === 0 || index == message.length-1 ? (
                          <strong>{point}</strong> 
                        ) : (
                        point
                        )}
                      </div>
                    ))}
                  </div>
                  <br /> <br />
                </div>
              </div>
            )}
          </div> 
        </div>
      )}
    </div>
  );
}

// import { useState } from "react";
// import axios from "axios";
// import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
// import "./styles.css";

// export default function App() {
//   const [file, setFile] = useState(null);
//   const [results, setResults] = useState([]);
//   const [frequentWords, setFrequentWords] = useState([]);
//   const [message, setMessage] = useState("");
//   const [predicted_class, setPredicted_class] = useState("");
//   const [loading, setLoading] = useState(false);
//   const [showResults, setShowResults] = useState(false);

//   const handleFileChange = (e) => setFile(e.target.files[0]);

//   const handleUpload = async () => {
//     if (!file) return alert("Please select a CSV file.");
  
//     setLoading(true);
//     const formData = new FormData();
//     formData.append("file", file);
  
//     try {
//       const response = await axios.post(
//         "http://localhost:8000/upload/", 
//         formData,
//         {
//           headers: { "Content-Type": "multipart/form-data" },
//           timeout: 120000  // 30-second timeout
//         }
//       );
      
//       // Ensure data exists before setting state
//         setResults(response.data.flagged_texts || []);
//         setFrequentWords(response.data.frequent_words || []);
//         setMessage(response.data.message || "");
//         setPredicted_class(response.data.predicted_class || "");
//     } catch (error) {
//       console.error("Full error:", error);
//       if (error.code === "ECONNABORTED") {
//         alert("Request timed out. Please try again.");
//       } else {
//         alert(`Error: ${error.response?.data?.detail || error.message}`);
//       }
//     } finally {
//       setLoading(false);
//     }
//   };

  // const handleShowResults = () => {
  //   setShowResults(true);
  // };

//   return (
//     <div className="container">
//       <h1 className="heading">🧠 Early Suicide & Depression Detection</h1>

//       <input
//         type="file"
//         accept=".csv"
//         onChange={handleFileChange}
//         className="file-input"
//       />
//       <button onClick={handleUpload} disabled={loading} className="upload-button">
//         {loading ? "Processing..." : "Upload & Analyze"}
//       </button>

      // {loading && (
      //   <div className="loading-screen">
      //     <div className="spinner"></div>
      //     <p>Analyzing your data...</p>
      //   </div>
      // )}

//       {results.length > 0 && !loading && (
//         <div className="results-container">
//           <h2 className="text-xl font-semibold mb-4">Flagged Messages 🚨</h2>
//           <table className="results-table">
//             <thead>
//               <tr>
//                 <th>Text (Preview)</th>
//                 <th>Suicidal Probability</th>
//               </tr>
//             </thead>
//             <tbody>
//               {results.map((item, index) => (
//                 <tr key={index}>
//                   <td>{item.text.substring(0, 100)}...</td>
//                   <td>{(item.probability * 100).toFixed(2)}%</td>
//                 </tr>
//               ))}
//             </tbody>
//           </table>

  //         <div className="chart-container">
  //           <h2 className="chart-title">Most Frequent Suicide-Related Words</h2>
  //           <ResponsiveContainer width="100%" height={350}>
  //             <BarChart data={frequentWords.map(([word, count]) => ({ word, count }))}>
  //               <XAxis dataKey="word" />
  //               <YAxis />
  //               <Tooltip />
  //               <Bar dataKey="count" fill="#8884d8" />
  //             </BarChart>
  //           </ResponsiveContainer>
            
  //           <button onClick={handleShowResults} className="results-button">
  //             Show Analysis Results
  //           </button>
            
  //           {showResults && (
  //             <div className="prediction-results">
  //               <h3>System Prediction: <span className="predicted-class">{predicted_class}</span></h3>
  //               <div className="message-container">
  //                 <h4>Recommendations:</h4>
  //                 {Array.isArray(message) ? (
  //                   <ul>
  //                     {message.map((point, index) => (
  //                       <li key={index}>{point}</li>
  //                     ))}
  //                   </ul>
  //                 ) : (
  //                   <p>{message}</p>
  //                 )}
  //               </div>
  //             </div>
  //           )}
  //          </div> 
  //       </div>
  //     )}
  //   </div>
  // );
// }

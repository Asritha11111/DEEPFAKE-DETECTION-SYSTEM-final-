const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const { PythonShell } = require('python-shell');
const fs = require('fs');

const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const { v4: uuidv4 } = require('uuid');
const ffmpeg = require('fluent-ffmpeg');
const ffmpegPath = require('ffmpeg-static');
ffmpeg.setFfmpegPath(ffmpegPath);
// User database file
const USERS_FILE = path.join(__dirname, 'users.json');
const JWT_SECRET = 'your-secret-key-change-this'; // Change to a strong secret later

// Helper functions to read/write users
function readUsers() {
  const data = fs.readFileSync(USERS_FILE, 'utf8');
  return JSON.parse(data);
}

function writeUsers(users) {
  fs.writeFileSync(USERS_FILE, JSON.stringify(users, null, 2));
}

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Configure file upload
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const uploadDir = 'uploads';
        if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + path.extname(file.originalname));
    }
});
const upload = multer({ storage: storage });
// Registration endpoint
app.post('/api/register', express.json(), async (req, res) => {
  const { name, email, password } = req.body;
  if (!name || !email || !password) {
    return res.status(400).json({ error: 'All fields required' });
  }
  const users = readUsers();
  if (users.find(u => u.email === email)) {
    return res.status(400).json({ error: 'Email already exists' });
  }
  const hashedPassword = await bcrypt.hash(password, 10);
  const user = { id: uuidv4(), name, email, password: hashedPassword };
  users.push(user);
  writeUsers(users);
  res.json({ message: 'User created' });
});

// Login endpoint
app.post('/api/login', express.json(), async (req, res) => {
  const { email, password } = req.body;
  const users = readUsers();
  const user = users.find(u => u.email === email);
  if (!user) return res.status(401).json({ error: 'Invalid credentials' });
  const valid = await bcrypt.compare(password, user.password);
  if (!valid) return res.status(401).json({ error: 'Invalid credentials' });
  const token = jwt.sign({ id: user.id, email: user.email }, JWT_SECRET, { expiresIn: '1d' });
  res.json({ token, user: { id: user.id, name: user.name, email: user.email } });
});

// Update user profile (name)
app.put('/api/user', authenticateToken, express.json(), (req, res) => {
  const { name } = req.body;
  if (!name) return res.status(400).json({ error: 'Name is required' });
  const users = readUsers();
  const userIndex = users.findIndex(u => u.id === req.user.id);
  if (userIndex === -1) return res.status(404).json({ error: 'User not found' });
  users[userIndex].name = name;
  writeUsers(users);
  res.json({ message: 'Profile updated', user: { id: users[userIndex].id, name: users[userIndex].name, email: users[userIndex].email } });
});

// Middleware to authenticate JWT tokens
function authenticateToken(req, res, next) {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];
  if (!token) return res.status(401).json({ error: 'Token required' });
  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) return res.status(403).json({ error: 'Invalid token' });
    req.user = user;
    next();
  });
}

// Prediction endpoint
app.post('/api/predict', authenticateToken, upload.single('image'), async (req, res) => {
  // ... rest of the code remains unchanged
      try {
        if (!req.file) {
            return res.status(400).json({ error: 'No image uploaded' });
        }
        
        const imagePath = path.join(__dirname, req.file.path);
        
        let options = {
            mode: 'text',
            pythonPath: 'python',
            scriptPath: __dirname,
            args: [imagePath]
        };
        
        PythonShell.run('predict.py', options).then(results => {
            const prediction = parseFloat(results[0]);
            const result = {
                prediction: prediction,
                is_real: prediction > 0.5,
                confidence: prediction > 0.5 ? prediction : 1 - prediction
            };
            res.json(result);
        }).catch(err => {
            console.error(err);
            res.status(500).json({ error: 'Prediction failed' });
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Server error' });
    }
});

app.post('/api/predict-video', authenticateToken, upload.single('video'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No video uploaded' });
  const videoPath = path.join(__dirname, req.file.path);
  const outputDir = path.join(__dirname, 'temp_frames_' + Date.now());
  fs.mkdirSync(outputDir, { recursive: true });

  const frameRate = 1; // frames per second
  try {
    await new Promise((resolve, reject) => {
      ffmpeg(videoPath)
        .outputOptions([`-vf fps=${frameRate}`])
        .output(path.join(outputDir, 'frame-%04d.jpg'))
        .on('end', resolve)
        .on('error', reject)
        .run();
    });

    const frames = fs.readdirSync(outputDir).filter(f => f.endsWith('.jpg')).sort();
    const results = [];

    for (const frame of frames) {
      const framePath = path.join(outputDir, frame);
      const prediction = await new Promise((resolve, reject) => {
        PythonShell.run('predict.py', {
          mode: 'text',
          pythonPath: 'python',
          scriptPath: __dirname,
          args: [framePath]
        }, (err, output) => {
          if (err) reject(err);
          else resolve(parseFloat(output[0]));
        });
      });
      results.push({ frame, confidence: prediction, is_fake: prediction < 0.5 });
    }

    const fakeFrames = results.filter(r => r.is_fake).length;
    const fakePercentage = (fakeFrames / results.length) * 100;
    const overallIsFake = fakePercentage > 50;

    // Clean up temporary files
    fs.rmSync(outputDir, { recursive: true, force: true });
    fs.unlinkSync(videoPath);

    res.json({
      overall: overallIsFake ? 'FAKE' : 'REAL',
      fake_percentage: fakePercentage,
      frames_analyzed: results.length,
      fake_frames: fakeFrames,
      results: results
    });
  } catch (err) {
    console.error('Video processing error:', err);
    if (fs.existsSync(outputDir)) fs.rmSync(outputDir, { recursive: true, force: true });
    if (fs.existsSync(videoPath)) fs.unlinkSync(videoPath);
    res.status(500).json({ error: 'Video processing failed' });
  }
});
// Audio deepfake detection
app.post('/api/predict-audio', authenticateToken, upload.single('audio'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No audio uploaded' });
  const audioPath = path.join(__dirname, req.file.path);
  const options = {
    mode: 'text',
    pythonPath: 'python',
    scriptPath: __dirname,
    args: [audioPath]
  };
  PythonShell.run('predict_audio.py', options).then(results => {
    const prediction = parseFloat(results[0]);
    fs.unlinkSync(audioPath);
    res.json({
      prediction: prediction,
      is_real: prediction > 0.5,
      confidence: prediction > 0.5 ? prediction : 1 - prediction
    });
  }).catch(err => {
    console.error(err);
    fs.unlinkSync(audioPath);
    res.status(500).json({ error: 'Audio prediction failed' });
  });
});
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
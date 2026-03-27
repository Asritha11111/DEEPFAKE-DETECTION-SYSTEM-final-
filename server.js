const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const { PythonShell } = require('python-shell');
const fs = require('fs');

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

// Prediction endpoint
app.post('/api/predict', upload.single('image'), async (req, res) => {
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

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
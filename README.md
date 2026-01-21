# COVID-19 Dashboard Web Application

A full-stack COVID-19 dashboard built using **Node.js**, **Express.js**, **MongoDB**, and **vanilla HTML/CSS/JavaScript**.  
The application provides user authentication, secure REST APIs, and interactive dashboards for analyzing COVID-19 data, comparison views, heatmaps, and predictive analytics.

---

## 📌 Features

### User Features
- User Signup & Login  
- JWT-based authentication  
- Interactive COVID-19 dashboard  
- Region-wise comparison  
- Heatmap visualization  
- Predictive analytics page  

### Backend Capabilities
- Modular MVC architecture  
- REST API with Express.js  
- MongoDB database integration  
- Secure password hashing  
- Protected routes using middleware  

---

## 🛠️ Tech Stack

**Frontend:**  
- HTML5  
- CSS3  
- JavaScript (Vanilla)

**Backend:**  
- Node.js  
- Express.js  
- MongoDB + Mongoose  
- JWT  
- bcryptjs  
- dotenv  
- CORS

---

## 📁 Project Structure

```
CovidDashboard/
│
├── middleware/
│   └── authMiddleware.js
│
├── models/
│   └── User.js
│
├── routes/
│   └── user.js
│
├── public/
│   ├── signup.html
│   ├── login.html
│   ├── dashboard.html
│   ├── comparison.html
│   ├── heatmap.html
│   ├── advanced_prediction.html
│   ├── style.css
│   ├── script.js
│   ├── dashboard.js
│   └── auth.js
│
├── server.js
├── package.json
└── .env
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository  
```bash
git clone https://github.com/your-username/covid-dashboard.git
cd covid-dashboard
```

### 2. Install dependencies  
```bash
npm install
```

### 3. Configure environment variables  
Create a `.env` file in the root directory:

```
PORT=5000
MONGO_URI=your_mongodb_connection_string
JWT_SECRET=your_secret_key
```

### 4. Start the application  
```bash
npm start
```

Application runs at:  
**http://localhost:5000**

---

## 🌐 Application Flow

- `/signup` → User registration  
- `/login` → User authentication  
- `/dashboard` → COVID-19 analytics  
- `/comparison` → Region comparison  
- `/heatmap` → Spread visualization  
- `/advanced_prediction` → Prediction dashboard  

---

## 🔒 Authentication Flow

- Passwords hashed with bcrypt  
- JWT token generated on login  
- Auth middleware protects secured routes  

---

## 📸 Screenshots (Aligned Gallery)

<p align="center">
  <img src="https://github.com/user-attachments/assets/8a2935aa-02c4-4392-a108-71f32c9dfa07" width="32%" />
  <img src="https://github.com/user-attachments/assets/7d214de4-e69a-4d42-bba7-6b0b71f283ae" width="32%" />
  <img src="https://github.com/user-attachments/assets/394f015c-cdc3-48a1-8491-6352b39d69c0" width="32%" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/612a4dac-045a-4700-b64c-4686b2cc9b17" width="32%" />
  <img src="https://github.com/user-attachments/assets/09c77d87-8ea6-4a33-b53c-a1a10b2cadbe" width="32%" />
  <img src="https://github.com/user-attachments/assets/16a68bc2-598c-4101-8007-740124d41d03" width="32%" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/aea22d4e-f3ac-4e37-8720-9b6132c32825" width="32%" />
  <img src="https://github.com/user-attachments/assets/e69391d6-8ce5-44e1-b876-8e94d882ee72" width="32%" />
  <img src="https://github.com/user-attachments/assets/fbd4b0d8-c9ea-4605-8909-237ac2748f42" width="32%" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/30d02c9f-4bcc-4fd2-bf87-1e94562e5cb5" width="32%" />
  <img src="https://github.com/user-attachments/assets/fff61c12-9c87-467d-a7a8-7a6d9245c547" width="32%" />
  <img src="https://github.com/user-attachments/assets/c2f95b07-2c8e-4088-b21d-9e788d946d50" width="32%" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/49324f3e-1fe2-4d15-b62e-91dee3830a9f" width="32%" />
  <img src="https://github.com/user-attachments/assets/9218ce48-7bba-4862-87f4-9d7aff125be0" width="32%" />
</p>

---

## 🚀 Future Enhancements
- Real-time data integration  
- Fully responsive interface  
- D3.js/Chart.js advanced visualizations  
- Role-based admin dashboard  
- Deployment on Render/AWS/Vercel  

---

## 👨‍💻 Author  
**Alex T Sabu**  
MSc Data Science  
Bengaluru, India  

---


# ============================================
# PROJECT 3: AI RECOMMENDATION LOGIC
# Tech Stack Recommender - WORKING VERSION
# DecodeLabs - Industrial Training Kit
# ============================================

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import threading

# ------------------------------
# DATASET: Job roles with required skills
# ------------------------------
data = {
    "job_role": [
        "Cloud Architect",
        "DevOps Engineer",
        "Machine Learning Engineer",
        "Backend Developer (Python)",
        "Full Stack Developer",
        "Data Engineer",
        "Security Analyst",
        "Frontend Developer",
        "AI Research Scientist",
        "Automation Engineer",
        "Mobile Developer",
        "Game Developer",
        "Database Administrator",
        "Network Engineer",
        "QA Automation Engineer"
    ],
    "skills": [
        "AWS, Cloud Computing, Terraform, Docker, Kubernetes, Python, Linux",
        "CI/CD, Jenkins, Docker, Kubernetes, Ansible, AWS, Linux, Git",
        "Python, TensorFlow, PyTorch, Scikit-learn, SQL, Data Processing",
        "Python, Django, Flask, PostgreSQL, REST APIs, Git, Docker",
        "JavaScript, React, Node.js, MongoDB, HTML, CSS, Git",
        "SQL, Python, Spark, Hadoop, Airflow, ETL, Data Warehousing",
        "Security, Firewalls, SIEM, Python, Incident Response, Risk Assessment",
        "JavaScript, TypeScript, React, Vue.js, CSS, Webpack, Figma",
        "Python, Deep Learning, NLP, Computer Vision, Math, Research",
        "Python, Automation, Ansible, Scripting, CI/CD, Monitoring, Linux",
        "Kotlin, Swift, React Native, Flutter, Firebase, REST APIs",
        "C++, Unity, Unreal Engine, C#, 3D Math, Shaders",
        "SQL, Oracle, PostgreSQL, Query Optimization, Backup Recovery",
        "Cisco, Routing, Switching, Firewalls, TCP/IP, Wireshark",
        "Python, Selenium, JUnit, TestNG, CI/CD, Jenkins"
    ]
}

df = pd.DataFrame(data)

# Preprocess skills
def preprocess_skills(skill_string):
    cleaned = re.sub(r'[,\n]', ' ', skill_string.lower())
    cleaned = ' '.join(cleaned.split())
    return cleaned

df["cleaned_skills"] = df["skills"].apply(preprocess_skills)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df["cleaned_skills"])

# ------------------------------
# RECOMMENDATION ENGINE CLASS
# ------------------------------
class TechStackRecommender:
    def __init__(self, job_df, tfidf_mat, vectorizer_obj):
        self.df = job_df
        self.tfidf_matrix = tfidf_mat
        self.vectorizer = vectorizer_obj
        self.last_scores = None

    def recommend(self, user_skills, top_n=5):
        if not user_skills:
            sim_scores = np.array(self.tfidf_matrix.sum(axis=1)).flatten()
            indices = np.argsort(sim_scores)[::-1][:top_n]
            recommendations = self.df.iloc[indices][["job_role", "skills"]].copy()
            recommendations["similarity_score"] = sim_scores[indices] / sim_scores.max()
            self.last_scores = sim_scores
            return recommendations

        user_skill_str = " ".join([s.strip().lower() for s in user_skills])
        user_vector = self.vectorizer.transform([user_skill_str])
        cosine_sim = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
        self.last_scores = cosine_sim
        sorted_indices = np.argsort(cosine_sim)[::-1][:top_n]
        
        recommendations = self.df.iloc[sorted_indices][["job_role", "skills"]].copy()
        recommendations["similarity_score"] = cosine_sim[sorted_indices]
        return recommendations

# ------------------------------
# MAIN GUI APPLICATION
# ------------------------------
class RecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DecodeLabs - AI Recommendation Engine")
        self.root.geometry("1200x750")
        self.root.configure(bg="#0a0e27")
        
        # Color scheme
        self.colors = {
            "bg": "#0a0e27",
            "card": "#141b33",
            "accent": "#6366f1",
            "accent_light": "#818cf8",
            "success": "#10b981",
            "text": "#f3f4f6",
            "text_secondary": "#9ca3af"
        }
        
        # Initialize recommender
        self.recommender = TechStackRecommender(df, tfidf_matrix, vectorizer)
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg=self.colors["bg"], height=120)
        header_frame.pack(fill=tk.X, padx=20, pady=(10, 0))
        header_frame.pack_propagate(False)
        
        title = tk.Label(
            header_frame,
            text="⚡ DECODELABS AI RECOMMENDATION ENGINE ⚡",
            font=("Segoe UI", 24, "bold"),
            fg=self.colors["text"],
            bg=self.colors["bg"]
        )
        title.pack(pady=(25, 5))
        
        subtitle = tk.Label(
            header_frame,
            text="Content-Based Filtering | TF-IDF Weighting | Cosine Similarity",
            font=("Segoe UI", 11),
            fg=self.colors["accent_light"],
            bg=self.colors["bg"]
        )
        subtitle.pack()
        
        # Main content area
        main_frame = tk.Frame(self.root, bg=self.colors["bg"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left Panel - Input
        left_panel = tk.Frame(main_frame, bg=self.colors["card"])
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Right Panel - Results
        right_panel = tk.Frame(main_frame, bg=self.colors["card"])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Create sections
        self.create_input_section(left_panel)
        self.create_results_section(right_panel)
        
        # Footer
        footer_frame = tk.Frame(self.root, bg=self.colors["bg"], height=40)
        footer_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        self.status_var = tk.StringVar()
        self.status_var.set("✅ System Ready | TF-IDF Vector Space Loaded")
        
        status_label = tk.Label(
            footer_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            fg=self.colors["text_secondary"],
            bg=self.colors["bg"]
        )
        status_label.pack()
        
        tk.Label(
            footer_frame,
            text="DecodeLabs AI | Project 3: Industrial Training Kit",
            font=("Segoe UI", 8),
            fg=self.colors["text_secondary"],
            bg=self.colors["bg"]
        ).pack(pady=(5, 0))
    
    def create_input_section(self, parent):
        inner = tk.Frame(parent, bg=self.colors["card"])
        inner.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        # Title
        tk.Label(
            inner,
            text="📝 USER SKILL INPUT",
            font=("Segoe UI", 16, "bold"),
            fg=self.colors["text"],
            bg=self.colors["card"]
        ).pack(anchor=tk.W, pady=(0, 15))
        
        # Instruction
        tk.Label(
            inner,
            text="Enter your technical skills separated by commas:",
            font=("Segoe UI", 10),
            fg=self.colors["text_secondary"],
            bg=self.colors["card"]
        ).pack(anchor=tk.W, pady=(0, 10))
        
        # Text entry
        self.skills_entry = tk.Text(
            inner,
            height=6,
            font=("Consolas", 11),
            bg="#1a2340",
            fg=self.colors["text"],
            insertbackground=self.colors["accent_light"],
            relief=tk.FLAT,
            padx=15,
            pady=15,
            wrap=tk.WORD
        )
        self.skills_entry.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Placeholder
        self.placeholder_text = "e.g., Python, Cloud Computing, Automation, Docker"
        self.skills_entry.insert("1.0", self.placeholder_text)
        self.skills_entry.bind("<FocusIn>", self.on_entry_focus_in)
        self.skills_entry.bind("<FocusOut>", self.on_entry_focus_out)
        
        # Buttons
        btn_frame = tk.Frame(inner, bg=self.colors["card"])
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.example_btn = tk.Button(
            btn_frame,
            text="📋 Load Example",
            command=self.load_example,
            font=("Segoe UI", 10, "bold"),
            bg="#1a2340",
            fg=self.colors["accent_light"],
            cursor="hand2",
            relief=tk.FLAT,
            padx=20,
            pady=10
        )
        self.example_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.recommend_btn = tk.Button(
            btn_frame,
            text="🚀 GENERATE RECOMMENDATIONS",
            command=self.get_recommendations,
            font=("Segoe UI", 11, "bold"),
            bg=self.colors["accent"],
            fg="white",
            cursor="hand2",
            relief=tk.FLAT,
            padx=20,
            pady=10
        )
        self.recommend_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Loading indicator
        self.loading_label = tk.Label(
            inner,
            text="",
            font=("Segoe UI", 10),
            fg=self.colors["accent_light"],
            bg=self.colors["card"]
        )
        self.loading_label.pack(pady=(10, 0))
    
    def create_results_section(self, parent):
        inner = tk.Frame(parent, bg=self.colors["card"])
        inner.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        # Title
        tk.Label(
            inner,
            text="🎯 RECOMMENDED CAREER PATHS",
            font=("Segoe UI", 16, "bold"),
            fg=self.colors["text"],
            bg=self.colors["card"]
        ).pack(anchor=tk.W, pady=(0, 15))
        
        # Canvas with scrollbar
        canvas_frame = tk.Frame(inner, bg=self.colors["card"])
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_canvas = tk.Canvas(
            canvas_frame,
            bg=self.colors["card"],
            highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.results_canvas.yview)
        self.results_canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.results_frame = tk.Frame(self.results_canvas, bg=self.colors["card"])
        self.results_canvas.create_window((0, 0), window=self.results_frame, anchor=tk.NW)
        
        self.results_frame.bind("<Configure>", self.on_frame_configure)
        self.results_canvas.bind("<Configure>", self.on_canvas_configure)
        self.results_canvas.bind("<MouseWheel>", self.on_mousewheel)
    
    def on_entry_focus_in(self, event):
        if self.skills_entry.get("1.0", tk.END).strip() == self.placeholder_text:
            self.skills_entry.delete("1.0", tk.END)
            self.skills_entry.config(fg=self.colors["text"])
    
    def on_entry_focus_out(self, event):
        if not self.skills_entry.get("1.0", tk.END).strip():
            self.skills_entry.insert("1.0", self.placeholder_text)
            self.skills_entry.config(fg=self.colors["text_secondary"])
    
    def load_example(self):
        self.skills_entry.delete("1.0", tk.END)
        self.skills_entry.insert("1.0", "Python, Cloud Computing, Automation, Docker")
        self.skills_entry.config(fg=self.colors["text"])
        
        # Flash effect
        original_bg = self.skills_entry.cget("bg")
        self.skills_entry.config(bg="#2d3a5e")
        self.root.after(200, lambda: self.skills_entry.config(bg=original_bg))
    
    def get_recommendations(self):
        # Get input
        raw_input = self.skills_entry.get("1.0", tk.END).strip()
        if raw_input == self.placeholder_text or not raw_input:
            messagebox.showwarning("Input Required", "Please enter your technical skills.")
            return
        
        skills_list = [s.strip() for s in raw_input.split(",") if s.strip()]
        
        if not skills_list:
            messagebox.showwarning("Invalid Input", "Please enter valid skills separated by commas.")
            return
        
        # Show loading
        self.recommend_btn.config(state=tk.DISABLED, text="⏳ PROCESSING...")
        self.loading_label.config(text="🟡 Computing similarity scores...")
        self.status_var.set("🟡 Calculating TF-IDF vectors and cosine similarity...")
        
        # Run in thread
        thread = threading.Thread(target=self.process_recommendations, args=(skills_list,))
        thread.daemon = True
        thread.start()
    
    def process_recommendations(self, skills_list):
        import time
        time.sleep(0.3)
        
        recommendations = self.recommender.recommend(skills_list, top_n=5)
        
        # Update UI
        self.root.after(0, lambda: self.display_results(recommendations, skills_list))
    
    def display_results(self, recommendations, user_skills):
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Show user skills
        skills_frame = tk.Frame(self.results_frame, bg=self.colors["card"])
        skills_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(
            skills_frame,
            text=f"📊 Analyzing: {', '.join(user_skills)}",
            font=("Segoe UI", 10, "bold"),
            fg=self.colors["success"],
            bg=self.colors["card"]
        ).pack(anchor=tk.W)
        
        # Display each recommendation
        for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
            self.create_result_card(row, idx)
        
        # Reset UI
        self.recommend_btn.config(state=tk.NORMAL, text="🚀 GENERATE RECOMMENDATIONS")
        self.loading_label.config(text="")
        self.status_var.set("✅ Recommendations generated successfully!")
        
        # Scroll to top
        self.results_canvas.yview_moveto(0)
    
    def create_result_card(self, row, rank):
        # Card frame
        card = tk.Frame(
            self.results_frame,
            bg="#1a2340",
            relief=tk.FLAT
        )
        card.pack(fill=tk.X, pady=(0, 10))
        
        inner = tk.Frame(card, bg="#1a2340")
        inner.pack(padx=20, pady=15, fill=tk.X)
        
        # Header with rank
        header_frame = tk.Frame(inner, bg="#1a2340")
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        medals = {1: "🏆", 2: "🥈", 3: "🥉", 4: "📌", 5: "⭐"}
        medal = medals.get(rank, "📌")
        
        rank_label = tk.Label(
            header_frame,
            text=f"{medal} #{rank}",
            font=("Segoe UI", 14, "bold"),
            fg=self.colors["accent"],
            bg="#1a2340"
        )
        rank_label.pack(side=tk.LEFT, padx=(0, 15))
        
        title_label = tk.Label(
            header_frame,
            text=row["job_role"],
            font=("Segoe UI", 14, "bold"),
            fg=self.colors["text"],
            bg="#1a2340"
        )
        title_label.pack(side=tk.LEFT)
        
        # Score
        score_pct = row["similarity_score"] * 100
        
        score_frame = tk.Frame(inner, bg="#1a2340")
        score_frame.pack(fill=tk.X, pady=(5, 8))
        
        score_text = tk.Label(
            score_frame,
            text=f"Match Confidence: {score_pct:.1f}%",
            font=("Segoe UI", 10, "bold"),
            fg=self.colors["success"],
            bg="#1a2340"
        )
        score_text.pack(anchor=tk.W)
        
        # Progress bar
        progress_canvas = tk.Canvas(
            score_frame,
            height=8,
            bg="#0a0e27",
            highlightthickness=0
        )
        progress_canvas.pack(fill=tk.X, pady=(5, 0))
        
        # Draw progress
        progress_width = int((score_pct / 100) * 500)
        progress_canvas.create_rectangle(0, 0, progress_width, 8, fill=self.colors["accent"], outline="")
        
        # Skills
        skills_label = tk.Label(
            inner,
            text=f"🔧 Required Skills: {row['skills']}",
            font=("Segoe UI", 9),
            fg=self.colors["text_secondary"],
            bg="#1a2340",
            wraplength=400,
            justify=tk.LEFT
        )
        skills_label.pack(anchor=tk.W, pady=(5, 0))
    
    def on_frame_configure(self, event):
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
    
    def on_canvas_configure(self, event):
        self.results_canvas.itemconfig(1, width=event.width)
    
    def on_mousewheel(self, event):
        self.results_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

# ------------------------------
# MAIN ENTRY POINT
# ------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = RecommendationApp(root)
    root.mainloop()
import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# -------------------- Ensure NLTK Resources --------------------
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# -------------------- Load Model & Vectorizer --------------------
model = joblib.load('news_category_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# -------------------- Page Style --------------------
st.markdown("""
<style>
.stApp {
    background-image: linear-gradient(rgba(0, 0, 180, 0.5), rgba(0, 0, 180, 0.5)),
                      url("https://www.shutterstock.com/shutterstock/videos/3822164211/thumb/1.jpg?ip=x480");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: white;
}
h1 { color: #FFD700; text-align: center; }

/* Text area & Button styling */
textarea {
    border: 2px solid #002366 !important;
    border-radius: 10px !important;
}
div.stButton > button:first-child {
    background-color: #002366;
    color: white;
    border-radius: 10px;
    border: 1px solid white;
    padding: 0.6em 1.2em;
}
div.stButton > button:hover {
    background-color: #003cb3;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Sidebar Style --------------------
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: rgba(0, 70, 30, 0.45);
    color: white;
}
[data-testid="stSidebar"] h1, h2, h3 { color: #00171F; }
::-webkit-scrollbar-thumb { background: #00cfff; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# -------------------- Sidebar: Developer Info --------------------
with st.sidebar.expander("👨‍💻 Developer's Intro"):
    st.markdown("- **Hi, I'm Rayyan Ahmed**")
    st.markdown("- **Google Certified AI Prompt Specialist**")
    st.markdown("- **IBM Certified Advanced LLM FineTuner**")
    st.markdown("- **Google Certified Soft Skill Professional**")
    st.markdown("- **Hugging Face Certified: Fundamentals of LLMs**")
    st.markdown("- **Expert in EDA, ML, RL, ANN, CNN, CV, RNN, NLP, LLMs**")
    st.markdown("[💼 Visit LinkedIn](https://www.linkedin.com/in/rayyan-ahmed-504725321/)")

# -------------------- Sidebar: Tech Stack --------------------
with st.sidebar.expander("🛠️ Tech Stack Used"):
    st.markdown("""
    - **Python Libraries:** NumPy, Pandas, Matplotlib, Seaborn  
    - **NLP:** NLTK (Stopwords, Lemmatization)  
    - **Feature Engineering:** TF-IDF (Scikit-learn)  
    - **Model:** Logistic Regression (Scikit-learn)  
    - **Serialization:** Joblib  
    - **Web App:** Streamlit  
    - **Visualization:** WordCloud, Matplotlib  
    - **Deployment:** Git, Streamlit Community Cloud
    """)

# -------------------- Sidebar: Dataset Info --------------------
with st.sidebar.expander("🗂️ Trained Dataset Info"):
    st.markdown(f"- **Total Rows:** 120000")
    st.markdown(f"- **Total Columns:** 3")
    st.markdown(f"- **Categories:** 4 unique labels")

# -------------------- Sidebar: Model Performance --------------------
with st.sidebar.expander("📈 Model Performance"):
    st.markdown("**Model Used:** Logistic Regression (Multiclass)**")
    model_accuracy, precision, recall, f1 = 0.92, 0.91, 0.92, 0.91
    st.markdown(f"- **Accuracy:** `{model_accuracy * 100:.2f}%`")
    st.markdown(f"- **Precision:** `{precision * 100:.2f}%`")
    st.markdown(f"- **Recall:** `{recall * 100:.2f}%`")
    st.markdown(f"- **F1-Score:** `{f1 * 100:.2f}%`")
    st.markdown("""
    **Insights:**  
    - Model performs well across all four categories.  
    - TF-IDF + Logistic Regression gives strong baseline results.  
    - Can be enhanced using XGBoost or Neural Networks.
    """)

# -------------------- Main App --------------------
st.title("📰 News Category Classifier")
st.markdown("")
st.write("**Rayyan Ahmed | Elevvo**") 

st.markdown("""
### 🏷️ Available Categories:
1️⃣ **World** 🌍 2️⃣ **Sports** 🏅 3️⃣ **Business** 💼 4️⃣ **Science & Technology** 💻
""")

# Example inputs for easy testing
st.info("""
**🧠 Try These Headlines:**
- *Poland and Ukraine hold talks to establish stronger relations* → 🌍 **World**  
- *Lionel Messi leads Argentina to victory in World Cup qualifier* → 🏅 **Sports**  
- *Apple’s quarterly profits exceed Wall Street expectations* → 💼 **Business**  
- *NASA develops new AI system for satellite communication* → 💻 **Science & Tech**
""")

# -------------------- Prediction + History --------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = pd.DataFrame(columns=["Input Text", "Predicted Category"])

user_input = st.text_area("✍️ Enter a news headline or description:")

if st.button("🔍 Predict Category"):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
        return ' '.join(words)

    clean_text = preprocess_text(user_input)
    vectorized = tfidf.transform([clean_text])
    prediction = model.predict(vectorized)[0]

    category_names = {0: 'World 🌍', 1: 'Sports 🏅', 2: 'Business 💼', 3: 'Science & Tech 💻'}

    st.success(f"✅ **Predicted Category:** {category_names[prediction]}")

    # Save in session history
    st.session_state.prediction_history = pd.concat([
        st.session_state.prediction_history,
        pd.DataFrame({"Input Text": [user_input], "Predicted Category": [category_names[prediction]]})
    ], ignore_index=True)

# -------------------- Display & Download History --------------------
if not st.session_state.prediction_history.empty:
    st.markdown("---")
    st.subheader("📊 Prediction History")
    st.dataframe(st.session_state.prediction_history.tail(5))

    csv = st.session_state.prediction_history.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Prediction Report (CSV)",
        data=csv,
        file_name="news_predictions.csv",
        mime="text/csv"
    )

# -------------------- WordCloud Visualization --------------------
# -------------------- WordCloud Visualization --------------------
st.markdown("---")
st.subheader("🌈 WordCloud Visualization by Category")

# Try using test.csv if train.csv is not available
try:
    df = pd.read_csv('test.csv')
    st.info("Using test.csv for WordCloud visualization.")
except:
    st.error("⚠️ Could not find dataset file. Please upload one.")

category_labels = {1: 'World 🌍', 2: 'Sports 🏅', 3: 'Business 💼', 4: 'Science & Tech 💻'}

selected_category = st.selectbox(
    "Select a category to visualize most frequent words:",
    options=list(category_labels.keys()),
    format_func=lambda x: category_labels[x]
)

if 'df' in locals():
    if st.button("☁️ Generate WordCloud"):
        st.info(f"Generating WordCloud for **{category_labels[selected_category]}** ...")

        text_data = " ".join(df[df['Class Index'] == selected_category]['Description'].astype(str))

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(text_data)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)

        st.success(f"✅ WordCloud generated for {category_labels[selected_category]}")



